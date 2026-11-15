"""Recompute pointwise 95% paired hierarchical-bootstrap intervals.

Run with Python 3: python recompute_intervals.py
Uses only the standard library. Binary pass/fail labels are read as supplied;
this script does not re-execute generated code or verify those labels.
"""
import collections
import json
import random
from pathlib import Path

REPLICATES = 5000
SEED = 42
ROOT = Path(__file__).resolve().parent


def percentile(values, probability):
    values = sorted(values)
    position = (len(values) - 1) * probability
    lower = int(position)
    upper = min(lower + 1, len(values) - 1)
    return values[lower] + (position - lower) * (values[upper] - values[lower])


def evaluate(path):
    if '_multipl_e.' in path.name:
        return evaluate_multipl_e(path)
    cells = collections.defaultdict(lambda: collections.defaultdict(dict))
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        assert isinstance(row['passed'], bool)
        cell = cells[row['group']][row['task_id']]
        assert row['sample_idx'] not in cell, 'Duplicate sample index'
        cell[row['sample_idx']] = row['passed']
    assert set(cells) == {'GPT5-style', 'SyntaxBPE'}
    tasks = sorted(cells['SyntaxBPE'])
    assert set(tasks) == set(cells['GPT5-style'])
    benchmark = path.name.split('_', 2)[2].removesuffix('.answers.jsonl')
    expected_tasks = {'humaneval_plus': 164, 'mbpp': 500, 'mbpp_plus': 378}[benchmark]
    assert len(tasks) == expected_tasks
    arrays = {}
    for group in cells:
        arrays[group] = []
        for task in tasks:
            assert set(cells[group][task]) == set(range(10))
            arrays[group].append([cells[group][task][i] for i in range(10)])
    ours, baseline = arrays['SyntaxBPE'], arrays['GPT5-style']
    denominator = len(tasks) * 10
    ours_score = sum(map(sum, ours)) / denominator * 100
    baseline_score = sum(map(sum, baseline)) / denominator * 100
    rng = random.Random(SEED)
    differences = []
    for _ in range(REPLICATES):
        # Pair only the task resample; independently resample completions.
        task_indices = rng.choices(range(len(tasks)), k=len(tasks))
        delta = 0
        for i in task_indices:
            delta += sum(rng.choices(ours[i], k=10))
            delta -= sum(rng.choices(baseline[i], k=10))
        differences.append(delta / denominator * 100)
    return {
        'file': path.name,
        'tasks': len(tasks),
        'syntaxbpe_pass_at_1': ours_score,
        'gpt5_style_pass_at_1': baseline_score,
        'difference_pp': ours_score - baseline_score,
        'ci_lower_pp': percentile(differences, 0.025),
        'ci_upper_pp': percentile(differences, 0.975),
        'replicates': REPLICATES,
        'seed': SEED,
    }


def evaluate_multipl_e(path):
    cells = collections.defaultdict(lambda: collections.defaultdict(lambda: collections.defaultdict(dict)))
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        assert isinstance(row['passed'], bool)
        cell = cells[row['group']][row['language']][row['problem_id']]
        assert row['sample_idx'] not in cell, 'Duplicate model/language/problem/sample'
        cell[row['sample_idx']] = row['passed']
    assert set(cells) == {'GPT5-style', 'SyntaxBPE'}
    languages = sorted(cells['SyntaxBPE'])
    assert set(languages) == set(cells['GPT5-style'])
    assert set(languages) == {'cpp', 'cs', 'go', 'java', 'js', 'php', 'python', 'rust', 'ts'}
    problems = sorted({task for lang in languages for task in cells['SyntaxBPE'][lang]})
    assert len(problems) == 164
    paired = [[] for _ in problems]
    language_scores = {}
    tasks_per_language = {}
    for j, language in enumerate(languages):
        assert set(cells['SyntaxBPE'][language]) == set(cells['GPT5-style'][language])
        tasks_per_language[language] = len(cells['SyntaxBPE'][language])
        scores = {}
        for group in cells:
            for task, outcomes in cells[group][language].items():
                assert set(outcomes) == set(range(10))
            scores[group] = sum(sum(v.values()) for v in cells[group][language].values()) / (tasks_per_language[language] * 10) * 100
        language_scores[language] = scores
        for i, problem in enumerate(problems):
            if problem in cells['SyntaxBPE'][language]:
                ours = [cells['SyntaxBPE'][language][problem][k] for k in range(10)]
                baseline = [cells['GPT5-style'][language][problem][k] for k in range(10)]
                paired[i].append((j, ours, baseline))
    ours_score = sum(v['SyntaxBPE'] for v in language_scores.values()) / len(languages)
    baseline_score = sum(v['GPT5-style'] for v in language_scores.values()) / len(languages)
    rng = random.Random(SEED)
    differences = []
    for _ in range(REPLICATES):
        # Keep all available translations of a resampled problem together.
        task_indices = rng.choices(range(len(problems)), k=len(problems))
        deltas = [0] * len(languages)
        counts = [0] * len(languages)
        for i in task_indices:
            for j, ours, baseline in paired[i]:
                deltas[j] += sum(rng.choices(ours, k=10)) - sum(rng.choices(baseline, k=10))
                counts[j] += 10
        assert all(counts), 'Empty language in bootstrap replicate'
        differences.append(sum(d / n for d, n in zip(deltas, counts)) / len(languages) * 100)
    return {
        'file': path.name,
        'tasks': len(problems),
        'tasks_per_language': tasks_per_language,
        'language_scores': language_scores,
        'syntaxbpe_pass_at_1': ours_score,
        'gpt5_style_pass_at_1': baseline_score,
        'difference_pp': ours_score - baseline_score,
        'ci_lower_pp': percentile(differences, 0.025),
        'ci_upper_pp': percentile(differences, 0.975),
        'replicates': REPLICATES,
        'seed': SEED,
    }


if __name__ == '__main__':
    import argparse
    from concurrent.futures import ProcessPoolExecutor
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workers', type=int, default=1)
    args = parser.parse_args()
    assert args.workers >= 1
    paths = sorted((ROOT / 'answers').glob('*.jsonl'))
    if args.workers == 1:
        for path in paths:
            print(json.dumps(evaluate(path)), flush=True)
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            for result in pool.map(evaluate, paths):
                print(json.dumps(result), flush=True)
