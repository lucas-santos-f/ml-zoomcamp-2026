# Student dropout: flag at-risk students after the first semester

Midterm project for Machine Learning Zoomcamp 2026.

## Problem

About a third of the students in this dataset never finish their degree. Universities
usually notice when the student has already stopped coming. By then a tutor, a payment plan
or a change of course is too late.

The model here runs once the first semester's grades are in. For each student it gives the
probability of dropping out and a yes/no flag. The flag is tuned to catch 80% of the
students who will actually leave, so a student support office can call those students
during the second semester instead of finding out at the end of the year.

Only data known at that point is used: the application, the family background, the first
semester's results and the economy that year. The dataset also has second semester results,
and they would make the model look better, but they arrive too late to help.

## Data

*Predict Students' Dropout and Academic Success*, UCI Machine Learning Repository, id 697
(Realinho, Vieira Martins, Machado and Baptista, 2021), CC BY 4.0. 4,424 students from a
Portuguese polytechnic, enrolled 2008-2019, 36 columns plus the outcome (Dropout, Enrolled,
Graduate). Committed here as `data/students.csv`; the original is at
https://archive.ics.uci.edu/dataset/697.

Target: `dropout` = 1 for Dropout (32%), 0 for Graduate and Enrolled.

Two features are added in `prep.py`: the share of first semester units the student passed
(`approved_ratio_1st_sem`) and the share they sat an exam for (`evaluated_ratio_1st_sem`).

## What the notebook shows

`notebook.ipynb`: cleaning (no missing values, no duplicates), EDA and model selection.

- Tuition in arrears: 87% drop out, against 25% of students who are up to date.
  Debtors 62% vs 28%. Scholarship holders 12% vs 39%.
- By course, from 15% (nursing) to above 50% (equinculture, informatics engineering,
  evening management).
- `approved_ratio_1st_sem` has the strongest correlation with dropout, -0.59. Age at
  enrollment is next on the positive side, 0.25.
- Among the categorical columns, application mode and course carry the most mutual
  information with the target.

Models, AUC on the validation set (60/20/20 split, stratified):

| model | tuned | best | AUC |
|---|---|---|---|
| Logistic regression | C | C = 0.5 | 0.905 |
| Decision tree | max_depth, min_samples_leaf | depth 10, leaf 50 | 0.876 |
| Random forest | max_depth, min_samples_leaf, n_estimators | depth 15, leaf 1, 300 trees | 0.892 |
| XGBoost | eta, max_depth, min_child_weight, early stopping | eta 0.03, depth 4, mcw 1, 492 rounds | **0.908** |

XGBoost is used. On the test set it gets AUC 0.901. With a cut-off of 0.29 it flags 37% of
students and catches 228 of the 284 who dropped out; 69% of the flagged students really
leave.

## Files

```
data/students.csv     the dataset
prep.py               column names, added features, feature lists (shared by everything)
notebook.ipynb        EDA and model selection
train.py              trains the final model, saves model.bin
predict.py            FastAPI service, POST /predict
test.py               sends two example students to the service
Dockerfile
k8s/                  deployment and service for Kubernetes
pyproject.toml, uv.lock
```

## Running it

Dependencies are managed with [uv](https://docs.astral.sh/uv/) (`pip install uv` if you do
not have it). `uv sync` creates `.venv` with the exact versions from `uv.lock`.

```bash
uv sync
uv run python train.py          # prints the test AUC and writes model.bin
uv run python predict.py        # service on http://localhost:9696
uv run python test.py           # in a second terminal
```

To use the environment without `uv run`, activate it: `.venv\Scripts\activate` on Windows,
`source .venv/bin/activate` elsewhere.

To re-run the notebook: `uv run jupyter notebook notebook.ipynb`.

Example output of `test.py`:

```
at risk student: {'dropout_probability': 0.9584, 'at_risk': True}
same student, on track: {'dropout_probability': 0.0344, 'at_risk': False}
```

The first student passed 1 of 6 units and is behind on tuition; the second is the same
student with all units passed, fees paid and a scholarship.

### Docker

```bash
docker build -t student-dropout:v1 .
docker run -it --rm -p 9696:9696 student-dropout:v1
uv run python test.py
```

### Kubernetes (local, with kind)

```bash
kind create cluster --name ml-dropout
kind load docker-image student-dropout:v1 --name ml-dropout
kubectl apply -f k8s/
kubectl rollout status deployment/student-dropout
kubectl port-forward service/student-dropout 9697:80
uv run python test.py http://localhost:9697/predict
kind delete cluster --name ml-dropout
```

Run on 20 October 2026:

```
$ kubectl get pods,svc
NAME                                   READY   STATUS    RESTARTS   AGE
pod/student-dropout-6fd47bd87d-l4zz7   1/1     Running   0          12s
pod/student-dropout-6fd47bd87d-t5zgw   1/1     Running   0          12s

NAME                      TYPE        CLUSTER-IP     EXTERNAL-IP   PORT(S)   AGE
service/student-dropout   ClusterIP   10.96.225.66   <none>        80/TCP    12s

$ uv run python test.py http://localhost:9697/predict
at risk student: {'dropout_probability': 0.9584, 'at_risk': True}
same student, on track: {'dropout_probability': 0.0344, 'at_risk': False}
```

## Request format

`POST /predict` with the student's columns as in the source file, snake_case. Codes
(course, qualifications, occupations, application mode) are the integer codes from the
UCI documentation; `course` 9119 is informatics engineering. Columns with a default in
`predict.py` can be left out.

## Limitations

- One institution, one country. The course and family codes do not translate to other
  universities without retraining.
- *Enrolled* students count as not dropped out, but some of them will still leave.
- The cut-off of 0.29 was set on the test set to hit 80% recall. A support office with a
  different budget should move it: the notebook has the precision/recall table.
