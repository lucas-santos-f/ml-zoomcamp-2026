import sys

import requests

url = sys.argv[1] if len(sys.argv) > 1 else 'http://localhost:9696/predict'

# a first-year informatics student who passed 1 of 6 units and is behind on tuition
student = {
    'course': 9119, 'previous_qualification_grade': 128.0, 'admission_grade': 121.4,
    'mothers_qualification': 19, 'fathers_qualification': 37, 'mothers_occupation': 9,
    'fathers_occupation': 7, 'displaced': 0, 'debtor': 1, 'tuition_fees_up_to_date': 0,
    'gender': 1, 'scholarship_holder': 0, 'age_at_enrollment': 24,
    'curricular_units_1st_sem_enrolled': 6, 'curricular_units_1st_sem_evaluations': 8,
    'curricular_units_1st_sem_approved': 1, 'curricular_units_1st_sem_grade': 10.5,
    'unemployment_rate': 10.8, 'inflation_rate': 1.4, 'gdp': 1.74,
}
print('at risk student:', requests.post(url, json=student).json())

student.update({'debtor': 0, 'tuition_fees_up_to_date': 1, 'scholarship_holder': 1,
                'curricular_units_1st_sem_approved': 6, 'curricular_units_1st_sem_grade': 14.2})
print('same student, on track:', requests.post(url, json=student).json())
