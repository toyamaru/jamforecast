import pandas as pd
import datetime
import lightgbm as lgb
import jpholiday
from sklearn.preprocessing import LabelEncoder

class ScoringService(object):
    @classmethod
    def get_model(cls, model_path, inference_df, inference_log):
        #cls.model = lgb.Booster(model_file=f'{model_path}/model.txt')
        cls.model = lgb.Booster(model_file='../model/model.txt')
        cls.data = inference_df
        cls.log_pathes = inference_log

        return True


    @classmethod
    def predict(cls, input, input_log):
        def calc_holiday(ymd, t):
            if jpholiday.is_holiday(ymd + datetime.timedelta(days=t)):
                h = 1
            else:
                h = 0
            return h
        
        input.fillna(0, inplace=True)
        input_ = input.sort_values(['start_code', 'end_code', 'datetime']).reset_index(drop=True)
        
        input_['year'] = input_['datetime'].apply(lambda x: x.year)
        input_['month'] = input_['datetime'].apply(lambda x: x.month)
        input_['day'] = input_['datetime'].apply(lambda x: x.day)
        input_['dayofweek'] = input_['datetime'].apply(lambda x: x.weekday())
        input_['hour'] = input_['datetime'].apply(lambda x: x.hour)
        
        input_date = pd.DataFrame(input_['datetime'].drop_duplicates().reset_index(drop=True))
        input_date.columns = ['date_unique']
        input_date['holiday_t-1'] = input_date.apply(lambda x: calc_holiday(x['date_unique'], -1), axis=1)
        input_date['holiday_t+0'] = input_date.apply(lambda x: calc_holiday(x['date_unique'], 0), axis=1)
        input_date['holiday_t+1'] = input_date.apply(lambda x: calc_holiday(x['date_unique'], 1), axis=1)
        input_ = input_.merge(input_date, left_on='datetime', right_on='date_unique', how='left')
        input_.drop('date_unique', axis=1, inplace=True)
        
        input_['rain_t'] = 0
        input_['rain_k'] = 0
        
        input_['section'] = input_['start_code'].astype(str) + '_' + input_['KP'].astype(str) + '_' + input_['end_code'].astype(str)
        input_['start_code_'] = input_['start_code']
        input_['end_code_'] = input_['end_code']
        cat_cols = ['road_code', 'start_code', 'end_code', 'section', 'direction', 'hour', 'dayofweek']
        num_cols = ['year', 'month', 'day', 'hour', 'search_specified', 'search_unspecified', 
        'KP', 'start_KP', 'end_KP', 'limit_speed', 'OCC', 'holiday_t-1', 'holiday_t+0', 'holiday_t+1', 'rain_t', 'rain_k']
        feature_cols = cat_cols + num_cols
        
        le_dict = {}
        for c in cat_cols:
            le = LabelEncoder()
            input_[c] = le.fit_transform(input_[c])
            le_dict[c] = le

        prediction = input_[feature_cols]
        prediction_predict = pd.DataFrame(cls.model.predict(prediction))
        prediction_predict.index = input_.index
        
        prediction = pd.concat([input_[['datetime', 'start_code_', 'end_code_', 'KP']], prediction_predict.astype(int)], axis=1)
        prediction.rename(columns={0: 'prediction', 'start_code_': 'start_code', 'end_code_': 'end_code'}, inplace=True)
        #prediction.to_csv("result0.csv")
        #prediction = prediction[['datetime', 'start_code', 'end_code', 'KP', 'prediction']]

        return prediction
