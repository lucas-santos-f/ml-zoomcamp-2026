# Question 3: load the frozen pipeline and score one lead.
# Run it from the homework directory so model.py and pipeline.bin are next to it:
#   uv run python q3_predict.py

import pickle

from model import normalize_record

lead = {
    "lead_source": "paid_ads",
    "industry": "technology",
    "employment_status": "employed",
    "location": "north_america",
    "number_of_courses_viewed": 2,
    "annual_income": 79276.0,
    "interaction_count": 4,
    "lead_score": 0.41,
}

with open("pipeline.bin", "rb") as f:
    pipeline = pickle.load(f)

probability = pipeline.predict_proba([normalize_record(lead)])[0, 1]
print(round(probability, 3), "(raw:", probability, ")")
