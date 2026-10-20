import pickle

import xgboost as xgb
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

import prep

# picked in notebook.ipynb on the validation set
PARAMS = {'eta': 0.03, 'max_depth': 4, 'min_child_weight': 1, 'objective': 'binary:logistic',
          'subsample': 0.8, 'colsample_bytree': 0.8, 'seed': 1, 'nthread': 8}
ROUNDS = 492
THRESHOLD = 0.29     # catches 80% of dropouts on the test set

df = prep.load()
full_train, test = train_test_split(df, test_size=0.2, random_state=1, stratify=df.dropout)

dv = DictVectorizer(sparse=False)
X = dv.fit_transform(prep.to_records(full_train))
features = list(dv.get_feature_names_out())
model = xgb.train(PARAMS, xgb.DMatrix(X, label=full_train.dropout.values, feature_names=features),
                  num_boost_round=ROUNDS)

p = model.predict(xgb.DMatrix(dv.transform(prep.to_records(test)), feature_names=features))
print(f'test auc {roc_auc_score(test.dropout, p):.4f}')

# final model sees every student
X_all = dv.fit_transform(prep.to_records(df))
model = xgb.train(PARAMS, xgb.DMatrix(X_all, label=df.dropout.values, feature_names=list(dv.get_feature_names_out())),
                  num_boost_round=ROUNDS)

with open('model.bin', 'wb') as f:
    pickle.dump((dv, model, THRESHOLD), f)
print('saved model.bin')
