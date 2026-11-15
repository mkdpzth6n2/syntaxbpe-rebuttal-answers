# Corrected confidence intervals and per-task outcomes

We have corrected an implementation issue in the confidence-interval calculation. The table below reports the updated intervals for MultiPL-E, HE+, MBPP, and MBPP+.

Each cell reports the SyntaxBPE-minus-GPT-5-style pass@1 difference in percentage points, followed by its pointwise 95% paired hierarchical-bootstrap confidence interval.

| Stage | Size | MultiPL-E Avg. | HE+ | MBPP | MBPP+ |
|---|---|---|---|---|---|
| Base | 1B | +1.9 [+0.6, +3.1] | +3.5 [+1.3, +5.8] | +1.6 [-0.4, +3.6] | +1.7 [-0.5, +3.9] |
| Base | 3B | +2.0 [+0.6, +3.4] | +1.5 [-0.8, +4.0] | +3.2 [+0.8, +5.5] | +1.6 [-0.4, +3.7] |
| SFT | 1B | +2.8 [+1.0, +4.7] | +2.3 [-1.6, +6.3] | +1.8 [-0.8, +4.6] | +3.0 [-0.5, +6.6] |
| SFT | 3B | +2.3 [+0.9, +3.9] | +1.6 [-2.1, +5.4] | +1.8 [-0.5, +4.2] | +2.2 [-0.1, +4.5] |

## Reproducing the intervals

Run `python recompute_intervals.py` from this repository, or add `--workers 4` to process datasets in parallel without changing their results. The [script](recompute_intervals.py) uses 5,000 bootstrap replicates and random seed 42 independently for each dataset. Each replicate resamples tasks with replacement, sharing the task resample between models, and independently resamples the 10 binary completion outcomes within each sampled model–task cell. The bounds are the 2.5th and 97.5th percentiles, using linear interpolation; the table rounds to one decimal place.

For MultiPL-E, the outer sampling unit is the underlying HumanEval problem (`problem_id`). All available language translations of a sampled problem are kept together. Completions are independently resampled within each model–language–problem cell, and the unweighted mean of the nine language-level pass@1 scores is recomputed in every replicate. Some problems lack translations in some languages; each language's denominator is recomputed from its available sampled translations. The script reads the supplied pass/fail labels and does not re-execute generated code.

## Per-task outcomes

The files contain both models' generated completions and pass/fail outcomes. HE+ contains 164 tasks, MBPP contains 500 tasks, and MBPP+ contains 378 tasks, with 10 completions per model per task. Their records are identified by `group`, `task_id`, and `sample_idx`.

MultiPL-E contains 164 underlying problems and 1,432 language-specific tasks per model, with 10 completions per model–language–problem cell. Language task counts are: C++ 161, C# 158, Go 154, Java 158, JavaScript 161, PHP 161, Python 164, Rust 156, and TypeScript 159. Use `group`, `language`, `problem_id`, and `sample_idx` to identify a MultiPL-E record; `task_id` alone does not distinguish languages.

| Stage | Size | MultiPL-E outcomes | HE+ outcomes | MBPP outcomes | MBPP+ outcomes |
|---|---|---|---|---|---|
| Base | 1B | [JSONL](answers/base_1b_multipl_e.answers.jsonl) | [JSONL](answers/base_1b_humaneval_plus.answers.jsonl) | [JSONL](answers/base_1b_mbpp.answers.jsonl) | [JSONL](answers/base_1b_mbpp_plus.answers.jsonl) |
| Base | 3B | [JSONL](answers/base_3b_multipl_e.answers.jsonl) | [JSONL](answers/base_3b_humaneval_plus.answers.jsonl) | [JSONL](answers/base_3b_mbpp.answers.jsonl) | [JSONL](answers/base_3b_mbpp_plus.answers.jsonl) |
| SFT | 1B | [JSONL](answers/sft_1b_multipl_e.answers.jsonl) | [JSONL](answers/sft_1b_humaneval_plus.answers.jsonl) | [JSONL](answers/sft_1b_mbpp.answers.jsonl) | [JSONL](answers/sft_1b_mbpp_plus.answers.jsonl) |
| SFT | 3B | [JSONL](answers/sft_3b_multipl_e.answers.jsonl) | [JSONL](answers/sft_3b_humaneval_plus.answers.jsonl) | [JSONL](answers/sft_3b_mbpp.answers.jsonl) | [JSONL](answers/sft_3b_mbpp_plus.answers.jsonl) |
