# ZeroSCROLLS

This repository contains code to run inference on the [ZeroSCROLLS](https://www.zero.scrolls-benchmark.com/) benchmark.

## Setup

* Install [torch](https://pytorch.org/get-started/locally/).
* Install transformers
* pip install -r requirements.txt


## Load the data
- via [🤗 Datasets (huggingface/datasets)](https://huggingface.co/datasets/tau/zero_scrolls/viewer/book_sum_sort/test) library (recommended):
```python
from datasets import load_dataset

gov_report = load_dataset("tau/zero_scrolls", "gov_report")["test"]
"""
Options are: ["gov_report", "summ_screen_fd", "qmsum", "squality", "qasper","narrative_qa", "quality", "musique", "space_digest","book_sum_sort"]
"""
```

- via ZIP files, where each split is in a JSONL file:
  - [GovReport](https://zero-scrolls-tau.s3.us-east-2.amazonaws.com/gov_report.zip)
  - [SummScreenFD](https://zero-scrolls-tau.s3.us-east-2.amazonaws.com/summ_screen_fd.zip)
  - [QMSum](https://zero-scrolls-tau.s3.us-east-2.amazonaws.com/qmsum.zip)
  - [SQuALITY](https://zero-scrolls-tau.s3.us-east-2.amazonaws.com/squality.zip)
  - [Qasper](https://zero-scrolls-tau.s3.us-east-2.amazonaws.com/qasper.zip)
  - [NarrativeQA](https://zero-scrolls-tau.s3.us-east-2.amazonaws.com/narrative_qa.zip)
  - [QuALITY](https://zero-scrolls-tau.s3.us-east-2.amazonaws.com/quality.zip)
  - [MuSiQue](https://zero-scrolls-tau.s3.us-east-2.amazonaws.com/musique.zip)
  - [SpaceDigest](https://zero-scrolls-tau.s3.us-east-2.amazonaws.com/space_digest.zip)
  - [BookSumSort](https://zero-scrolls-tau.s3.us-east-2.amazonaws.com/book_sum_sort.zip)


## Inference with Huggingface models 
```bash
python experiments/hf/run_hf_model.py --model-name=google/flan-t5-small
```

Supported models:
* google/flan-t5-small
* google/flan-t5-base
* google/flan-t5-large
* google/flan-t5-xl
* google/flan-t5-xxl
* google/flan-ul2
* bigscience/T0pp

To add new models:
* Add them to `model_to_max_input_tokens` in [experiments/hf/run_hf_model.py]((https://github.com/tau-nlp/scrolls/tree/main/baselines))
* Make sure to load them with the appropriate architecture (i.e. modify the model initialization from T5ForConditionalGeneration in the same file, if needed)

## Inference with APIs
To run with models used in the [paper](https://arxiv.org/pdf/2305.14196.pdf)*:

```bash
# if you want to use openai models
export OPENAI_API_KEY=<insert token here> 
export OPENAI_ORG=<insert org here>

# if you want to use anthropic models
export ANTHROPIC_API_KEY=<insert token here>

# if you want to limit the number of examples to run per task
export MAX_EXAMPLES=10

python experiments/api/run_api_model.py --model_name=gpt-3.5-turbo --limit_to_n_examples=$MAX_EXAMPLES
```
*These models and APIs tend to update, see the paper for the versions used in the baselines.

Models supported:
* text-davinci-003
* gpt-3.5-turbo
* gpt-4
* claude-v1

To add new a new API, you need to:
* Implement a new class the inherits from [APIRunner](https://github.com/tau-nlp/zero_scrolls/blob/main/experiments/api/api.py#L16).
* Working examples for OpenAI and Anthropic APIs can be found in [openai_api.py](https://github.com/tau-nlp/zero_scrolls/blob/main/experiments/api/openai_api.py) and [anthropic_api.py](https://github.com/tau-nlp/zero_scrolls/blob/main/experiments/api/anthropic_api.py)

When using prompt with opening xml tags, (e.g. "... Assistant: &lt;answer&gt;") you should postprocess generations to remove the closing xml tags generated by the model before submitting.

## Prepare submission
To create a CSV file in the correct format for a leaderboard submission we recommend using our conversion script, [prepare_submission.py](https://github.com/tau-nlp/zero_scrolls/blob/main/prepare_submission.py).

Its inputs:

For each task, the predictions should be in a JSON file that is a mapping from an ID to a textual prediction:
```JSON
{
    "example_id1": "prediction1",
    "example_id2": "prediction2",
    ...
}
```
Please set:
* `{dataset_name}_PREDS_FILE` to be the path to a JSON file in the format above containing your predictions for `{dataset_name}`.
* `OUTPUT_DIR` to be the path you want the submission file will be saved to.

Run:
```bash
python prepare_submission.py \
--gov_report_file GOV_REPORT_PREDS_FILE \
--summ_screen_file SUMM_SCREEN_FD_PREDS_FILE \
--qmsum_file QMSUM_PREDS_FILE \
--squality_file SQUALITY_PREDS_FILE \
--qasper_file QASPER_PREDS_FILE \
--narrative_qa_file NARRATIVE_QA_PREDS_FILE \
--quality_file QUALITY_PREDS_FILE \
--musique_file MUSIQUE_PREDS_FILE \
--space_digest_file SPACE_DIGEST_PREDS_FILE \
--book_sum_sort_file BOOK_SUM_SORT_PREDS_FILE \
--output_dir OUTPUT_DIR
```

## Leaderboard
The live leaderboard is [here](https://www.zero.scrolls-benchmark.com/leaderboard). 



## Citation
```
@misc{shaham2023zeroscrolls,
      title={ZeroSCROLLS: A Zero-Shot Benchmark for Long Text Understanding}, 
      author={Uri Shaham and Maor Ivgi and Avia Efrat and Jonathan Berant and Omer Levy},
      year={2023},
      eprint={2305.14196},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```
If you find the ZeroSCROLLS data useful, please make sure to cite also the original dataset papers: [[bibtex]](https://zero-scrolls-tau.s3.us-east-2.amazonaws.com/zero_scrolls_datasets.bib)
