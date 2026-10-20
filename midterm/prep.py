"""Shared data preparation for the notebook, train.py and predict.py."""
import pandas as pd

COURSES = {
    33: 'biofuel_production', 171: 'animation_multimedia', 8014: 'social_service_evening',
    9003: 'agronomy', 9070: 'communication_design', 9085: 'veterinary_nursing',
    9119: 'informatics_engineering', 9130: 'equinculture', 9147: 'management',
    9238: 'social_service', 9254: 'tourism', 9500: 'nursing', 9556: 'oral_hygiene',
    9670: 'advertising_marketing', 9773: 'journalism', 9853: 'basic_education',
    9991: 'management_evening',
}

# integer codes in the source file that are categories, not quantities
CATEGORICAL = ['marital_status', 'application_mode', 'course', 'previous_qualification',
               'nacionality', 'mothers_qualification', 'fathers_qualification',
               'mothers_occupation', 'fathers_occupation']

# known when the first semester ends; the second semester columns are left out on purpose,
# the point is to flag students early enough to do something about it
NUMERICAL = ['application_order', 'daytime_evening_attendance', 'previous_qualification_grade',
             'admission_grade', 'displaced', 'educational_special_needs', 'debtor',
             'tuition_fees_up_to_date', 'gender', 'scholarship_holder', 'age_at_enrollment',
             'international',
             'curricular_units_1st_sem_credited', 'curricular_units_1st_sem_enrolled',
             'curricular_units_1st_sem_evaluations', 'curricular_units_1st_sem_approved',
             'curricular_units_1st_sem_grade', 'curricular_units_1st_sem_without_evaluations',
             'unemployment_rate', 'inflation_rate', 'gdp',
             'approved_ratio_1st_sem', 'evaluated_ratio_1st_sem']


def clean_names(df):
    df.columns = (df.columns.str.strip().str.lower()
                  .str.replace("'", '').str.replace(r'[^a-z0-9]+', '_', regex=True).str.strip('_'))
    return df


def add_features(df):
    df = df.copy()
    enrolled = df.curricular_units_1st_sem_enrolled
    df['approved_ratio_1st_sem'] = (df.curricular_units_1st_sem_approved / enrolled).where(enrolled > 0, 0)
    df['evaluated_ratio_1st_sem'] = (df.curricular_units_1st_sem_evaluations / enrolled).where(enrolled > 0, 0)
    df['course'] = df.course.map(COURSES).fillna(df.course.astype(str))
    for c in CATEGORICAL:
        df[c] = df[c].astype(str)
    return df


def load(path='data/students.csv'):
    df = clean_names(pd.read_csv(path, sep=';'))
    df = add_features(df)
    df['dropout'] = (df.target == 'Dropout').astype(int)
    return df


def to_records(df):
    return df[CATEGORICAL + NUMERICAL].to_dict(orient='records')
