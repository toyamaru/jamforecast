import datetime
import os

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, classification_report
from sklearn.model_selection import StratifiedGroupKFold
import time
import xgboost as xgb


SEED = 0

def f1(y_pred, dtrain):
    # y_pred = (y_pred > 0.5).astype(int)
    y_true = dtrain.get_label()
    score = f1_score(y_true, np.round(y_pred))
    return 'f1', score


def _train_xgb(X,
               Y,
               cv,
               model_path=None,
               params: dict=None,
               verbose: int=100):

    # パラメータがないときは、空の dict で置き換える
    if model_path is None:
        model_path = []
    if params is None:
        params = {}

    models = []
    n_records = len(X)
    # training data の target と同じだけのゼロ配列を用意
    oof_pred = np.zeros((n_records, ), dtype=np.float32)

    for i, (idx_train, idx_valid) in enumerate(cv):

        tmp_x_train, tmp_y_train = X[idx_train], Y[idx_train]
        x_train, y_train = [], []
        for x, y in zip(tmp_x_train.tolist(), tmp_y_train.tolist()):
            n = 7 if y == 1 else 1
            # if 213 <= x[8] <= 273:
            #     n = 8
            for _ in range(n):
                x_train.append(x)
                y_train.append(y)

        x_train = np.array(x_train)
        y_train = np.array(y_train)

        x_valid, y_valid = X[idx_valid], Y[idx_valid]

        # clf = lgb.LGBMClassifier(**params)
        clf = xgb.XGBClassifier(**params)
        # clf = xgb.XGBRegressor(**params)

        clf.fit(x_train, y_train,
                eval_set=[(x_valid, y_valid)],
                early_stopping_rounds=300, # 150,
                # categorical_feature=CAT_COLS,
                verbose=verbose,
                # eval_metric='binary:logistic')
                eval_metric='error@0.5')
        # eval_metric='logloss')
        # eval_metric=f1)

        pred_i = clf.predict(x_valid)
        oof_pred[idx_valid] = pred_i
        models.append(clf)

        print(f'fold{i + 1}')
        score = f1_score(y_valid, (pred_i > 0.5).astype(int))
        print(f' - valid - {score:.4f}')
        # ext_ids_valid = [ i for i in idx_valid if i in TEST_IDXS]
        # ext_ids_valid = [ i for i in idx_valid if i in TMP_TEST_IDXS]
        # score_test = f1_score(Y[ext_ids_valid], (oof_pred[ext_ids_valid] > 0.5).astype(int))
        # print(f' - test - {score_test:.4f}')
        print()

    score = f1_score(Y, (oof_pred > 0.5).astype(int))

    print('=' * 50)
    print(f'FINISH: CV Score:')
    print(f' - total - {score:.4f}')

    # score_tmp_test = f1_score(Y[TMP_TEST_IDXS], (oof_pred[TMP_TEST_IDXS] > 0.5).astype(int))
    # score_test = f1_score(Y[TEST_IDXS], (oof_pred[TEST_IDXS] > 0.5).astype(int))
    # print(f' - tmp - {score_tmp_test:.4f}')
    # print(f' - final - {score_test:.4f}')

    return score, oof_pred, models


def train(features_df: pd.DataFrame, feature_cols: list):

    # param 3 best
    params = {
        # 'objective': 'binary:logistic', # reg:logistic
        'objective': 'reg:logistic', #
        'booster': 'gbtree',
        # 'max_leaves':
        # 'num_leaves' : int(0.7 * 2 ** max_depth), #240
        'n_estimators': 6_000,
        'lambda': 0.0009109444365205688,
        'alpha': 0.039662902077230756,
        'subsample': 0.7541666880832332,
        'colsample_bytree': 0.8309999621210956,
        # 'max_delta_step': 4,
        'max_depth': 8,
        'min_child_weight': 2,
        'eta': 0.13694621326751983,
        'gamma': 8.484794735229636e-05,
        'grow_policy': 'depthwise',
        'enable_categorical': True,
        'base_score': 0.5,
        'importance_type': 'gain',
        'random_state': SEED,
        'n_jobs': os.cpu_count(),
        'gpu_id': 0,
        'tree_method': 'gpu_hist',
        'verbosity': 0,
    }


    # ========================================
    # train-validation split
    # ========================================
    target = 'is_congestion'
    n_split = 7

    kf = StratifiedGroupKFold(n_split) # , shuffle=True, random_state=SEED
    cv_list = list(kf.split(features_df, y=features_df[target], groups=features_df['weekofyear'] * 10_000 + features_df['dayofweek'] * 100 + features_df['hour']))

    # ========================================
    # define variables
    # ========================================
    X = features_df[feature_cols].values
    Y = features_df[target].values

    print(f'train shape: {features_df.shape}\n')
    # ========================================
    # training
    # ========================================

    t0 = time.time()

    score, preds, models = _train_xgb(X, Y=Y, params=params, cv=cv_list)

    print(f'Done. ({ time.time() - t0 :.3f}s) - { str(datetime.datetime.now())[:-3] }\n')

    print(classification_report(features_df['is_congestion'], preds))
    print(f"gt: {features_df['is_congestion'].sum()}")
    print(f"pred: {preds.sum()}")

    return models


if __name__ == '__main__':
    pass
