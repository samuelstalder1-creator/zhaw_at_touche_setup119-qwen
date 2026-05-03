# setup119-qwen Code Submission

This directory is a self-contained TIRA code submission for the
`advertisement-in-retrieval-augmented-generation-2026` task. The container
entrypoint is `/predict.py`.

At runtime the submission:

- loads the bundled local sklearn pipeline from `model/`
- reuses the input `qwen` field when present
- otherwise generates the neutral locally with `Qwen/Qwen2.5-1.5B-Instruct`
- embeds texts with `sentence-transformers/all-mpnet-base-v2`
- applies the saved `setup119` residual feature layout
- writes `predictions.jsonl` in the TIRA format

This package does not depend on a classifier hosted on Hugging Face or GitHub.
The trained classifier bundle is the local `model/embedding_lr_classifier.pkl`
plus `model/embedding_state.json`.

## Model Definition

Saved setup:

- trainer type: `embedding_residual_classifier`
- neutral field: `qwen`
- feature blocks: `[response - qwen]`
- threshold source: saved in `model/embedding_state.json`

## Submission Package Contents

- `predict.py`: runtime inference entrypoint used by TIRA
- `Dockerfile`: image definition used by `tira-cli code-submission`
- `requirements.txt`: Python dependencies installed into the container
- `.dockerignore`: excludes scratch files from the image context
- `model/`: local saved sklearn pipeline and embedding state
- `README.md`: submission specification and operator notes

## Runtime Contract

TIRA executes the submission with:

```bash
/predict.py
```

Supported inputs:

- `inputDataset`: mounted TIRA input directory
- `outputDir`: mounted TIRA output directory
- `--dataset`: TIRA dataset id, local directory, or local JSONL file
- `--input-directory`: explicit local or mounted input directory
- `--output-directory`: explicit output directory
- `--output`: explicit output file path

If the input is a directory, `predict.py` automatically discovers the most
likely response file by scanning for JSONL files whose rows contain at least
`id`, `query`, and `response`.

## Input Specification

Each input row must contain:

- `id`
- `query`
- `response`

Optional:

- `qwen`: precomputed neutral response

If `qwen` is missing or empty, the runtime generates it locally with Qwen and
caches one generated neutral per unique query during the run.

## Output Specification

The submission writes:

```text
predictions.jsonl
```

Each row has this shape:

```json
{"id":"7O2H5WQK-3656-2FVX","label":1,"tag":"zhawAtToucheSetup119"}
```

The default tag is derived from the saved state, so this package emits
`zhawAtToucheSetup119` unless you override `--tag`.

## Local Verification

Run on a local directory or JSONL file:

```bash
./predict.py \
  --dataset ../../data/task \
  --output ./out/predictions.jsonl
```

Run against a TIRA dataset id:

```bash
./predict.py \
  --dataset advertisement-in-retrieval-augmented-generation-2026/ads-in-rag-task-1-detection-spot-check-20260422-training \
  --output ./out/predictions.jsonl
```

The TIRA-style environment variables also work:

```bash
inputDataset=../../data/task outputDir=./out ./predict.py
```

Useful overrides:

```bash
./predict.py \
  --dataset ../../data/task \
  --output ./out/predictions.jsonl \
  --model-dir ./model \
  --qwen-model Qwen/Qwen2.5-1.5B-Instruct \
  --batch-size 32 \
  --max-length 512 \
  --threshold 0.6135369759433669 \
  --device cpu
```

## Validate The Docker Submission

Use this section before uploading to TIRA to validate that the Dockerized
submission behaves like a real TIRA run.

### Prerequisites

- Docker is installed and running
- `tira` is installed: `pip3 install tira`
- you are registered for the task in TIRA
- for real uploads, the git repository is clean: `git status`

Authenticate and verify the local TIRA client:

```bash
tira-cli login --token <YOUR_TIRA_TOKEN>
tira-cli verify-installation --task advertisement-in-retrieval-augmented-generation-2026
```

If you use Docker Desktop with the containerd image store enabled, TIRA may
reject uploaded images even though the local build and push succeed. In that
case, force Docker v2 manifest output during submission:

```bash
tira-cli code-submission \
  --path . \
  --task advertisement-in-retrieval-augmented-generation-2026 \
  --dataset ads-in-rag-task-1-detection-spot-check-20260422-training \
  --command '/predict.py' \
  --build-args '--output type=docker --provenance=false'
```

If Docker still exports an incompatible image, disable Docker Desktop's
`Use containerd for pulling and storing images` setting, rebuild, and retry the
submission.

If the failure happens before your submission image is built, `tira-cli` is
likely rejecting its own internal `tira-mini` preflight image before the
`--build-args` above are applied. In that case, prepend the repo-local Docker
wrapper so every `docker build` invoked by `tira-cli` gets the compatibility
flags, including the preflight check:

```bash
PATH="${PWD}/tools:${PATH}" tira-cli code-submission \
  --path . \
  --task advertisement-in-retrieval-augmented-generation-2026 \
  --dataset ads-in-rag-task-1-detection-spot-check-20260422-training \
  --command '/predict.py'
```

Build locally:

```bash
docker build -t zhaw-at-touche-setup119-qwen-local .
```

Dry-run through TIRA:

```bash
tira-cli code-submission \
  --dry-run \
  --path . \
  --task advertisement-in-retrieval-augmented-generation-2026 \
  --dataset ads-in-rag-task-1-detection-spot-check-20260422-training \
  --command '/predict.py'
```

What this validates:

- the Docker image builds successfully
- `/predict.py` starts correctly inside the container
- the runtime can read `$inputDataset`
- the runtime writes a valid JSONL prediction file to `$outputDir`
- the output format is acceptable for the task

The Docker build preloads both runtime transformer models:

- `sentence-transformers/all-mpnet-base-v2`
- `Qwen/Qwen2.5-1.5B-Instruct`

That keeps the final TIRA execution offline-safe while still using the local
`model/` classifier bundle.

## Submit To TIRA

From this directory, submit the package with:

```bash
tira-cli code-submission \
  --path . \
  --task advertisement-in-retrieval-augmented-generation-2026 \
  --dataset ads-in-rag-task-1-detection-spot-check-20260422-training \
  --command '/predict.py'
```
