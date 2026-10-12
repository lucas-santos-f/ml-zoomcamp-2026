# Homework 5 — Deploying Machine Learning Models

The 2026 cohort ships a frozen release for this one: the trained pipeline, the
training script, the API and the Dockerfile are all checked into the course repo
under `cohorts/2026/homework/05-deployment`. Nothing had to be trained here, so
this folder holds the one script I wrote plus the answers.

Artifact verified before anything else:

```
sha256sum pipeline.bin
1646bbdcd38d4f044da6b630c5b332c93a314245a8c21929011c42de51f629f1
```

## Answers

| Q | Answer |
|---|---|
| 1 | uv 0.12.18 (setup check, not graded) |
| 2 | scikit-learn 1.7.2 |
| 3 | 0.533 |
| 4 | 0.770 |
| 5 | python:3.11.15-slim-bookworm |
| 6 | 0.770 |

Q3 is the `paid_ads` lead scored from the pickle directly, 0.5329944186432507
before rounding. Q4 and Q6 are the `organic_search` lead, 0.769799, first
through the local uvicorn process and then through the container. The two
agreeing is the whole point of the exercise: same artifact, same answer, two
different runtimes.

Q2 comes from `pyproject.toml` and is confirmed in `uv.lock`, which pins
`scikit-learn==1.7.2` with `exclude-newer = 2026-09-14`. Q5 is read from the
`FROM` line, which pins the base image by digest as well as by tag.

## The checksum did not match at first

`/health` reports a sha256 for `feature_defaults.json`, and the homework asks to
check it against `model_metadata.json`. It did not match:

```
/health              5cb9195a7a160fdc91a1b8f57172055e33ced1994870742b937d5273f4021c67
model_metadata.json  1694f842b908fac9f53c0cc19be6d91f934befc63af23cc32fefafde69acd0cd
```

Git had rewritten the line endings on checkout. The file is 122 bytes with six
CRLF pairs on this machine; replacing them with LF reproduces the recorded hash
exactly. The model hash matched all along, because `pipeline.bin` is binary and
Git leaves it alone.

It changes no answer — `json.load` does not care about line endings — but it
does break the integrity check the exercise asks you to run, and on a different
day it would be the thing that quietly invalidates a signed artifact. Converting
the file back to LF and rebuilding makes both hashes match.

## Running it

```bash
cd repo/cohorts/2026/homework/05-deployment
uv sync --locked
uv run python smoke_test.py

cp ../../../../solucoes/hw05/q3_predict.py .
uv run python q3_predict.py                 # Q3

uv run uvicorn predict:app --host 0.0.0.0 --port 9696   # Q4

docker build -t zoomcamp-model:2026-hw5 .
docker run --rm -p 9696:9696 zoomcamp-model:2026-hw5
python q6_test.py                            # Q6
```
