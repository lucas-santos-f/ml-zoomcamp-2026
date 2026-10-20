import pickle

import pandas as pd
import xgboost as xgb
from fastapi import FastAPI
from pydantic import BaseModel

import prep

with open('model.bin', 'rb') as f:
    dv, model, threshold = pickle.load(f)

app = FastAPI(title='student-dropout')


class Student(BaseModel):
    # same names as the source file after prep.clean_names
    marital_status: int = 1
    application_mode: int = 1
    application_order: int = 1
    course: int
    daytime_evening_attendance: int = 1
    previous_qualification: int = 1
    previous_qualification_grade: float
    nacionality: int = 1
    mothers_qualification: int
    fathers_qualification: int
    mothers_occupation: int
    fathers_occupation: int
    admission_grade: float
    displaced: int
    educational_special_needs: int = 0
    debtor: int
    tuition_fees_up_to_date: int
    gender: int
    scholarship_holder: int
    age_at_enrollment: int
    international: int = 0
    curricular_units_1st_sem_credited: int = 0
    curricular_units_1st_sem_enrolled: int
    curricular_units_1st_sem_evaluations: int
    curricular_units_1st_sem_approved: int
    curricular_units_1st_sem_grade: float
    curricular_units_1st_sem_without_evaluations: int = 0
    unemployment_rate: float
    inflation_rate: float
    gdp: float


@app.get('/health')
def health():
    return {'status': 'ok'}


@app.post('/predict')
def predict(student: Student):
    df = prep.add_features(pd.DataFrame([student.model_dump()]))
    X = dv.transform(prep.to_records(df))
    p = float(model.predict(xgb.DMatrix(X, feature_names=list(dv.get_feature_names_out())))[0])
    return {'dropout_probability': round(p, 4), 'at_risk': p >= threshold}


if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=9696)
