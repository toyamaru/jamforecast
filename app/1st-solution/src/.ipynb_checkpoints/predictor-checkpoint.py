from glob import glob
import pickle

import pandas as pd
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb

import train

import pdb

TRIN_MODE = 'train'
PRED_MODE = 'pred'

class ScoringService(object):

    add_train_df = None
    # train_df = None
    models = None
    n_splits = 7
    congestion_count_df = None
    congestion_count_day_df = None
    label_encoder = None

    CAT_COLS = ['road_code', 'road_code_1', 'road_code_2', 'start_code', 'end_code', 'direction', 'dayofweek', 'weekofyear', 'dayofyear', 'year', 'month', 'day', 'is_pre_congestion']
    NUM_COLS = ['speed', 'OCC', 'allCars', 'search_1h', 'search_unspec_1d', 'hour', 'KP', 'start_KP', 'end_KP', 'limit_speed', 'start_lng', 'end_lng', 'congestion_count']
    FEATURE_COLS = CAT_COLS + NUM_COLS

    @classmethod
    def _expand_datetime(cls, df):
        if 'datetime' in df.columns:
            df['year'] = df['datetime'].dt.year
            df['month'] = df['datetime'].dt.month
            df['day'] = df['datetime'].dt.day
            df['dayofyear'] = df['datetime'].dt.dayofyear
            df['weekofyear'] = df['datetime'].dt.weekofyear
            df['dayofweek'] = df['datetime'].dt.dayofweek
            df['hour'] = df['datetime'].dt.hour

        return df

    @classmethod
    def _create_features(cls, raw_features_df, mode='pred') -> pd.DataFrame:

        # congestion_df = pd.concat([cls.train_df[['start_code', 'end_code', 'is_congestion']], raw_features_df[['start_code', 'end_code', 'is_congestion']]])
        # congestion_count_df = congestion_df.groupby(['start_code', 'end_code'], as_index=False).agg({ 'is_congestion': 'sum' }).rename(columns={ 'is_congestion': 'congestion_count'})

        features_df = cls._expand_datetime(raw_features_df)
        features_df['date'] = pd.to_datetime(features_df['datetime'], format='%Y-%m-%d')

        if mode == TRIN_MODE:
            cls.congestion_count_df = features_df.groupby(['start_code', 'end_code'], as_index=False).agg({ 'is_congestion': 'sum' }).rename(columns={ 'is_congestion': 'congestion_count'})
            # cls.congestion_count_day_df = features_df.groupby(['dayofyear'], as_index=False).agg({ 'is_congestion': 'sum' }).rename(columns={ 'is_congestion': 'congestion_day_count'})
        features_df = features_df.merge(cls.congestion_count_df, on=['start_code', 'end_code'], how='left')
        # features_df = features_df.merge(cls.congestion_count_day_df, on=['dayofyear'], how='left')

        if mode == TRIN_MODE:            
            next_train_df = features_df[['datetime', 'start_code', 'end_code', 'is_congestion']]
            features_df.rename(columns={ 'is_congestion': 'is_pre_congestion' }, inplace=True)        
            next_train_df['datetime'] -= pd.to_timedelta(1, 'd')
            features_df = features_df.merge(next_train_df, on=['datetime', 'start_code', 'end_code'], how='inner')
        elif mode == PRED_MODE:
            features_df.rename(columns={ 'is_congestion': 'is_pre_congestion' }, inplace=True)

        features_df['pred-datetime'] = features_df.datetime + pd.to_timedelta(1, 'd')

        features_df['road_code_1'] = 0
        features_df.loc[features_df.start_code < features_df.end_code, 'road_code_1'] = features_df.loc[features_df.start_code < features_df.end_code, 'end_code']
        features_df.loc[features_df.start_code > features_df.end_code, 'road_code_1'] = features_df.loc[features_df.start_code > features_df.end_code, 'start_code']
        features_df['road_code_2'] = features_df['start_code'] * 1_000_000 + features_df['end_code']

        if mode == TRIN_MODE:
            cls.label_encoder = {}
            for col in cls.CAT_COLS:
                le = LabelEncoder()
                le.fit(features_df[col])
                features_df[col] = le.transform(features_df[col])
                cls.label_encoder[col] = le
        elif mode == PRED_MODE:
            for col in cls.CAT_COLS:
                try:
                    le = cls.label_encoder[col]
                    features_df[col] = le.transform(features_df[col])
                except Exception as e:
                    print(f"{col}: {le.classes_}")
                    print(e) 

        return features_df

    @classmethod
    def merge_data(cls, train_df, road_df, search_data_df, search_unspec_data_df):
        # 欠損値の除外
        # train_df = train_df[train_df['speed'].isnull() == False]
        # road_df.dropna(inplace=True)
        # search_data_df.dropna(inplace=True)
        # search_unspec_data_df.dropna(inplace=True)

        # datetimeからdateを作成
        train_df['date'] = train_df['datetime'].apply(lambda x: x.split()[0])
        train_df['datetime'] = pd.to_datetime(train_df['datetime'])
        train_df['date'] = pd.to_datetime(train_df['date'])
        search_data_df['datetime'] = pd.to_datetime(search_data_df['datetime'])
        search_data_df['datetime'] -= pd.to_timedelta(1, 'd')
        search_unspec_data_df['date'] = pd.to_datetime(search_unspec_data_df['date'])
        search_unspec_data_df['date'] -= pd.to_timedelta(1, 'd')
        # search_unspec_data_df['date'] = search_unspec_data_df['date'].astype(str)

        # データのマージ
        train_df = train_df.merge(road_df, on=['start_code', 'end_code'], how='left')
        train_df = train_df.merge(search_data_df, on=['datetime', 'start_code', 'end_code'], how='left')
        train_df = train_df.merge(search_unspec_data_df, on=['date', 'start_code', 'end_code'], how='left')
        # train_df.sort_values(['date', 'start_code', 'end_code'], inplace=True)
        # train_df.reset_index(drop=True, inplace=True)
        # train_df.drop(columns='date', inplace=True)

        # データ型の変更
        # train_df['datetime'] = pd.to_datetime(train_df['datetime'])

        return train_df

    @classmethod
    def get_model(cls, model_path, inference_df):
        """Get model method

        Args:
            model_path (str): Path to the trained model directory.
            inference_df: Past data not subject to pred_df.

        Returns:
            bool: The return value. True for success.
        """
        cls.add_train_df = cls._create_features(inference_df.copy(), TRIN_MODE)
        train_df = pd.read_csv(f'{model_path}/train.csv')
        road_df = pd.read_csv(f'{model_path}/road.csv')
        search_data_df = pd.read_csv(f'{model_path}/search_data.csv')
        search_unspec_data_df = pd.read_csv(f'{model_path}/search_unspec_data.csv')
        features_df = cls.merge_data(train_df, road_df, search_data_df, search_unspec_data_df)
        # min_date = cls.add_train_df['date'].min()
        features_df = features_df[features_df.date < cls.add_train_df['date'].min()]
        features_df = pd.concat([features_df, cls.add_train_df])
        cls.models = train.train(features_df, cls.FEATURE_COLS)

        # cls.models = []
        # model = xgb.XGBClassifier()
        # for file_path in glob(f'{model_path}/*.json'):
        #     model.load_model(file_path)
        #     cls.models.append(model)
        #
        # cls.n_splits = len(cls.models)
        # cls.label_encoder = pickle.load(open(f'{model_path}/label_encoder.pkl', 'rb'))
        # cls.congestion_count_df = pickle.load(open(f'{model_path}/congestion_count_df.pkl', 'rb'))
        # cls.congestion_count_day_df = pickle.load(open(f'{model_path}/congestion_count_day_df.pkl', 'rb'))

        return True

    @classmethod
    def predict(cls, input):
        """Predict method

        Args:
            input: metadata of the sample you want to make inference from (DataFrame)

        Returns:
            results_df: Inference for the given input. Return columns must be ['datetime', 'start_code', 'end_code'](DataFrame).
        
        Tips:
            You can use past data by writing "cls.data".
        """

        features_df = cls._create_features(input.copy(), PRED_MODE)
        extract_features_df = features_df[cls.FEATURE_COLS]

        results_df = input[['datetime', 'start_code', 'end_code']]
        results_df['prediction'] = 0
        for i, model in enumerate(cls.models):
            results_df['prediction'] += model.predict(extract_features_df)

        results_df['prediction'] /= cls.n_splits
        results_df['prediction'] = results_df['prediction'].fillna(0).round().astype(int)

        return results_df
