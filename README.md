# Reproduction Guide

This folder hosts the scripts to run and post-process the SAT method across models and datasets.

## Files and roles
- `run_think_control_V5_qwen3.py`: run SAT on Qwen3 models for math datasets (`math500`, `gsm8k`, `aime2024`, `aime2025`, `amc`).
- `run_think_control_V5_llama.py`: SAT runner for the Llama family on the math datasets above.
- `run_think_control_V5_ds_qwen.py`: SAT runner for the ds_qwen family on the math datasets above.
- `run_think_control_V5_qwen3_gpqa.py`: SAT runner for Qwen3 models on the `gpqa` dataset.
- `run_think_control_V5_qwen3_human_eval.py`: SAT runner for Qwen3 models on the `human_eval` dataset.
- `run_think_control_V5_qwq.py`: SAT runner for QwQ models on the math datasets.
- `evaluate_r_detector_outputs.py`: evaluate the generated `r_*.jsonl` results against the gold answers (LLM-based grading; configure API/paths inside the script).
- `extract_item_done_data.py`: parse `r_*.jsonl` files to extract `item_done` events and token statistics (configure input/output paths inside the script).
- `tool.py`: lightweight utilities shared by the runners (feature definitions and z-score stats).

## Environment
The code is built on Python 3.10.16.
1) Install dependencies: `pip install -r requirements.txt`  
2) Run from the `Code` directory so relative paths resolve correctly.

## Running SAT (example)
Example: Qwen3-8B on `math500`, writing `r_math_Qwen3_8B_seed3407_sat.jsonl`.

```bash
python run_think_control_V5_qwen3.py \
  --model ../qwen3-8b \
  --input ../math500_test.jsonl \
  --out r_math_Qwen3_8B_seed3407_sat.jsonl
```

- `--model`: model checkpoint directory.
- `--input`: dataset JSONL (use the `<dataset>_test.jsonl` file, e.g., `math500_test.jsonl`, `gsm8k_test.jsonl`, `aime2024_test.jsonl`, `aime2025_test.jsonl`, `amc_test.jsonl`).
- `--out`: output result file (convention: `r_<dataset>_<Model>_seedXXXX_sat.jsonl`).

Switch scripts for other model families/datasets:
- Qwen3 math: `run_think_control_V5_qwen3.py`.
- QwQ math: `run_think_control_V5_qwq.py`.
- Llama math: `run_think_control_V5_llama.py`.
- Qwen3 gpqa: `run_think_control_V5_qwen3_gpqa.py` with `--input ../gpqa_test.jsonl`.
- Qwen3 human-eval: `run_think_control_V5_qwen3_human_eval.py` with the human-eval JSONL.

## Evaluating accuracy
`evaluate_r_detector_outputs.py` is currently configured via constants at the top (`API_KEY`, `BASE_URL`, `LLM_MODEL`, `INPUT_FILE`, `TEST_JSONL_FILE`, and output/checkpoint paths). To evaluate:
1) Replace `API_KEY` with your own key and set `INPUT_FILE` to the generated result file (e.g., `r_math_Qwen3_8B_seed3407_sat.jsonl`) and `TEST_JSONL_FILE` to the matching gold file (e.g., `math500_test.jsonl`).
2) Run `python evaluate_r_detector_outputs.py`.

> **Important Note on Accuracy:** > The automated evaluation script relies on LLM-based judging and may not be 100% accurate. For the most precise accuracy, we recommend manually verifying the model's raw outputs for cases marked as `is_correct: false`.  
> **(Please note that all experimental results reported in our paper have undergone manual verification to ensure correctness.)**

## Token statistics / item_done extraction
`extract_item_done_data.py` uses hard-coded paths near the top (`input_file`, `output_file`, `stats_file`, `json_output_file`). Point `input_file` to the result you want to inspect, then run:
```bash
python extract_item_done_data.py
```

## Notes
- All runners assume the dataset files follow the `<dataset>_test.jsonl` naming and live at the paths you pass via `--input`.
- Results `r_*.jsonl` files can be further analyzed with the evaluation and token-statistics scripts after editing their path/API settings.
