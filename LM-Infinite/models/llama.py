from typing import Optional, Tuple, Dict, Any
import math
import torch
import torch.nn.functional as F
from transformers import LlamaForCausalLM, LlamaTokenizer
from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding

from .model_base import Model_Base
from .lambda_attention import lambda_matmul

# ------------------------- Helpers -------------------------

def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    vec: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_ids: torch.LongTensor
) -> torch.Tensor:
    """Apply rotary positional embeddings to the input vector."""
    cos = cos.squeeze(0)
    sin = sin.squeeze(0)
    cos = cos[position_ids].unsqueeze(1)
    sin = sin[position_ids].unsqueeze(1)
    return vec * cos + rotate_half(vec) * sin


def detailed_lambda_attention(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    rot_query_states: torch.Tensor,
    rot_key_states: torch.Tensor,
    stationary_query_states: torch.Tensor,
    stationary_key_states: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    head_dim: int,
    device: torch.device,
    dtype: torch.dtype,
    q_len: int,
    kv_seq_len: int,
    global_branch: int,
    local_branch: int,
    limit_distance: Optional[int],
    triangle_offset: int,
    top_k_attention: Optional[int],
    top_k_insert_at: Optional[int],
    top_k_from_layer: Optional[int],
    top_k_to_layer: Optional[int],
    layer_i: int
) -> torch.Tensor:
    # rotary dot-products
    attn_weights = rot_query_states.matmul(rot_key_states.transpose(-1, -2)) / math.sqrt(head_dim)
    attn_stationary = stationary_query_states.matmul(stationary_key_states.transpose(-1, -2)) / math.sqrt(head_dim)

    if limit_distance is not None:
        attn_weights = attn_weights.triu(-local_branch + 1 + kv_seq_len - q_len)
        attn_weights += attn_stationary.tril(-limit_distance + kv_seq_len - q_len)

    if triangle_offset != 0:
        start = max(0, global_branch + local_branch - kv_seq_len + q_len)
        for i in range(start, q_len):
            col_hi = i - local_branch + 1 + kv_seq_len - q_len
            attn_weights[:, i, global_branch:col_hi] -= math.log(col_hi - global_branch) * triangle_offset

    if attention_mask is not None:
        if attention_mask.dim() == 4:
            attn_mask = attention_mask
        elif attention_mask.dim() == 3:
            attn_mask = attention_mask.unsqueeze(1)
        elif attention_mask.dim() == 2:
            attn_mask = attention_mask.unsqueeze(1).unsqueeze(1)
        else:
            raise ValueError(f"Unexpected attention mask shape: {attention_mask.shape}")
        attn_weights = attn_weights + attn_mask
        attn_weights = torch.max(attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min, device=device))

    if top_k_attention is not None:
        mask = torch.ones_like(attn_weights, dtype=torch.bool)
        mask = mask.tril(-limit_distance + kv_seq_len - q_len)
        mask[..., :global_branch] = False
        attn_weights.masked_fill_(mask, torch.finfo(attn_weights.dtype).min)

        i = q_len - 1
        col_hi = i - local_branch + 1 + kv_seq_len - q_len
        if top_k_from_layer <= layer_i < top_k_to_layer and col_hi >= global_branch + top_k_attention:
            near_q = query_states * cos[0, top_k_insert_at] + rotate_half(query_states) * sin[0, top_k_insert_at]
            near_attn = (near_q[..., i, None, :] @ stationary_key_states.transpose(-1, -2)) / math.sqrt(head_dim)
            slice_attn = near_attn[..., global_branch:col_hi]
            idx = torch.topk(slice_attn, top_k_attention, dim=-1)[1]
            mask2 = torch.ones_like(slice_attn, dtype=torch.bool)
            mask2.scatter_(-1, idx, False)
            attn_weights[..., i, global_branch:col_hi] = slice_attn.masked_fill(mask2, torch.finfo(slice_attn.dtype).min).squeeze(2)

    # stability
    probs = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(dtype)
    return probs @ value_states

# --------------------- Factory Patch ---------------------

def attn_forward_factory(
    self,
    use_lambda_mask: bool,
    local_branch: int,
    global_branch: int,
    limit_distance: Optional[int],
    triangle_offset: int,
    top_k_attention: Optional[int],
    top_k_insert_at: Optional[int],
    top_k_from_layer: Optional[int],
    top_k_to_layer: Optional[int],
    layer_i: int
):
    original_forward = self.forward

    def limited_distance_forward(
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        padding_mask: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        *args,
        **kwargs
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Patched forward for LM-Infinite λ-attention.
        Returns exactly two values: (output, attention_weights).
        """
        try:
            bsz, q_len, _ = hidden_states.size()
            head_dim = self.head_dim
            num_heads = self.config.num_attention_heads
            num_kv_heads = getattr(self.config, "num_key_value_heads", num_heads)
            group_size = num_heads // num_kv_heads if num_kv_heads > 0 else 1

            # QKV
            qs = self.q_proj(hidden_states).view(bsz, q_len, num_heads, head_dim).transpose(1, 2)
            ks = self.k_proj(hidden_states).view(bsz, q_len, num_kv_heads, head_dim).transpose(1, 2)
            vs = self.v_proj(hidden_states).view(bsz, q_len, num_kv_heads, head_dim).transpose(1, 2)
            if num_kv_heads < num_heads:
                ks = ks.repeat_interleave(group_size, dim=1)
                vs = vs.repeat_interleave(group_size, dim=1)

            dtype, device = qs.dtype, qs.device
            pkv = getattr(self, "past_key_value", past_key_value)
            if pkv is not None:
                if hasattr(pkv, "update"):
                    ks, vs = pkv.update(ks, vs, self.layer_idx, {})
                else:
                    ks = torch.cat([pkv[0], ks], dim=2)
                    vs = torch.cat([pkv[1], vs], dim=2)

            kv_len = ks.shape[-2]
            pos_ids = torch.arange(kv_len, device=device)[None]

            # rotary
            rotary_emb = getattr(self, "rotary_emb", None) or LlamaRotaryEmbedding(self.config, device=device)
            if hasattr(rotary_emb, "inv_freq"): rotary_emb.inv_freq = rotary_emb.inv_freq.to(torch.float32)
            cos, sin = rotary_emb(vs, pos_ids)

            # branches
            rqs = apply_rotary_pos_emb(qs, cos, sin, position_ids)
            rks = apply_rotary_pos_emb(ks, cos, sin, pos_ids)
            if limit_distance is None:
                sks, sqs = rks, rqs
            else:
                sks = ks
                eff = min(limit_distance, kv_len - 1)
                sqs = rqs * cos[0, eff] + rotate_half(rqs) * sin[0, eff]

            headwise = 33000
            # efficient lambda
            if use_lambda_mask and top_k_attention is not None:
                if q_len > headwise:
                    for h in range(num_heads):
                        qs[:, h, :-1] = (
                            lambda_matmul(rks[:, h], sks[:, h], rqs[:, h], sqs[:, h], local_branch, global_branch)
                            / math.sqrt(head_dim)
                        ).softmax().matmul(vs[:, h])[:, :-1]
                else:
                    qs[:, :, :-1] = (
                        lambda_matmul(rks, sks, rqs, sqs, local_branch, global_branch)
                        / math.sqrt(head_dim)
                    ).softmax().matmul(vs)[:, :, :-1]
                qs[:, :, -1] = detailed_lambda_attention(
                    qs[:, :, -1, None], ks, vs,
                    rqs[:, :, -1, None], rks,
                    sqs[:, :, -1, None], sks,
                    cos, sin,
                    None if attention_mask is None else attention_mask[:, :, -1, None],
                    head_dim, device, dtype,
                    1, kv_len,
                    global_branch, local_branch,
                    limit_distance, triangle_offset,
                    top_k_attention, top_k_insert_at,
                    top_k_from_layer, top_k_to_layer, layer_i
                ).squeeze(2)
            # lambda-only
            elif use_lambda_mask:
                if q_len > headwise:
                    for h in range(num_heads):
                        qs[:, h] = (
                            lambda_matmul(rks[:, h], sks[:, h], rqs[:, h], sqs[:, h], local_branch, global_branch)
                            / math.sqrt(head_dim)
                        ).softmax().matmul(vs[:, h])
                else:
                    qs = (
                        lambda_matmul(rks, sks, rqs, sqs, local_branch, global_branch)
                        / math.sqrt(head_dim)
                    ).softmax().matmul(vs)
            # fallback
            else:
                for h in range(num_heads):
                    qs[:, h] = detailed_lambda_attention(
                        qs[:, h], ks[:, h], vs[:, h],
                        rqs[:, h], rks[:, h],
                        sqs[:, h], sks[:, h], cos, sin, attention_mask,
                        head_dim, device, dtype, q_len, kv_len,
                        global_branch, local_branch,
                        limit_distance, triangle_offset,
                        top_k_attention, top_k_insert_at,
                        top_k_from_layer, top_k_to_layer, layer_i
                    )

            out = qs.transpose(1, 2).reshape(bsz, q_len, num_heads * head_dim)
            out = self.o_proj(out)

            attn_weights_out = None
            return out, attn_weights_out

        except Exception:
            # fallback two-value signature
            return original_forward(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
                padding_mask=padding_mask,
                cache_position=cache_position,
                *args, **kwargs
            )

    return limited_distance_forward
# ------------------- Model Class -------------------

class LLAMA_Model(Model_Base):
    """LLaMA model wrapper with LM-Infinite lambda-attention."""

    def __init__(self, model_name_or_path: str, tokenizer_path: str, max_length: int, truncation_side: str,
                 load_in_4bit: bool, device_map: Dict[str, Any],
                 use_lambda_mask: bool = True, local_branch: int = 1024, global_branch: int = 64,
                 limit_distance: int = 1024, triangle_offset: int = 0,
                 top_k_attention: Optional[int] = None, top_k_insert_at: Optional[int] = None,
                 top_k_from_layer: Optional[int] = None, top_k_to_layer: Optional[int] = None,
                 safe_mode: bool = True, debug: bool = False):
        super().__init__(max_length, truncation_side)
        # Load tokenizer & model
        self.tokenizer = LlamaTokenizer.from_pretrained(tokenizer_path)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.model = LlamaForCausalLM.from_pretrained(model_name_or_path,
                                                       load_in_4bit=load_in_4bit,
                                                       device_map=device_map)
        if debug:
            # Optional inspection
            from .llama import inspect_llama_model
            inspect_llama_model(self.model)
        # Store parameters
        self.use_lambda_mask = use_lambda_mask
        self.local_branch = local_branch
        self.global_branch = global_branch
        self.limit_distance = limit_distance
        self.triangle_offset = triangle_offset
        self.top_k_attention = top_k_attention
        self.top_k_insert_at = top_k_insert_at
        self.top_k_from_layer = top_k_from_layer
        self.top_k_to_layer = top_k_to_layer
        self.safe_mode = safe_mode
        # Patch attention
        for layer_i, layer in enumerate(self.model.model.layers):
            attn = layer.self_attn
            if not hasattr(attn, '_original_forward'):
                attn._original_forward = attn.forward
            layer.self_attn.forward = attn_forward_factory(
                attn, use_lambda_mask, local_branch, global_branch,
                limit_distance, triangle_offset,
                top_k_attention, top_k_insert_at,
                top_k_from_layer, top_k_to_layer,
                layer_i
            )
        if use_lambda_mask and not safe_mode:
            self.model.model._prepare_decoder_attention_mask = lambda *args, **kwargs: None

    def to(self, device: torch.device):
        if hasattr(self.model, 'to'):
            self.model.to(device)
        return self

    def generate(self, *args, **kwargs):
        return self.model.generate(*args, **kwargs)

# ------------------- Convenience Converter -------------------

def convert_llama_model(
    model: LlamaForCausalLM,
    local_branch: int = 1024,
    global_branch: int = 64,
    limit_distance: Optional[int] = None,
    triangle_offset: int = 0,
    safe_mode: bool = True,
    debug: bool = False
) -> LlamaForCausalLM:
    """
    Apply LM-Infinite lambda-attention patch to a pre-loaded LlamaForCausalLM.
    """
    # Optional debug inspection
    if debug:
        from .llama import inspect_llama_model
        inspect_llama_model(model)
    # Store original forwards if safe_mode
    if safe_mode:
        for layer in model.model.layers:
            attn = layer.self_attn
            if not hasattr(attn, '_original_forward'):
                attn._original_forward = attn.forward
    # Set config on all attn modules
    for layer in model.model.layers:
        layer.self_attn.config = model.config
    # Patch forwards
    for layer_i, layer in enumerate(model.model.layers):
        layer.self_attn.forward = attn_forward_factory(
            layer.self_attn, True, local_branch, global_branch,
            limit_distance or local_branch, triangle_offset,
            None, None, None, None,
            layer_i
        )
    return model