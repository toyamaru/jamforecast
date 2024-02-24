import datetime
import joblib
import jpholiday
import os
import pandas as pd

import consts

KEY = ['start_code', 'end_code']

class ScoringService(object):
    @classmethod
    def get_model(cls, model_path, inference_df):
        """Get model method

        Args:
            model_path (str): Path to the trained model directory.
            inference_df: Past data not subject to prediction.

        Returns:
            bool: The return value. True for success.
        """
        cls.data = inference_df
        cls.models = {}
        for k in range(1, consts.K + 1):
            cls.models[k] = joblib.load(os.path.join(model_path, f'model_{k}of{consts.K}.pkl'))

        return True

    @classmethod
    def load_past_data(cls, df: pd.DataFrame) -> dict:
        """_summary_

        Args:
            df (pd.DataFrame): _description_

        Returns:
            dict: _description_
        """
        past_data = {}
        for i in range(1, consts.use_past_days + 1):
            d = cls.today - datetime.timedelta(days=i)
            if os.path.exists(f'past_data{d.strftime("%Y-%m-%d")}.pkl'):
                past_data[i] = pd.read_pickle(f'past_data{d.strftime("%Y-%m-%d")}.pkl')
            else:
                past_df = df[['datetime'] + KEY + consts.past_cols].copy()
                past_df['hour'] = past_df['datetime'].dt.hour
                past_data[i] = past_df[['hour'] + KEY + consts.past_cols]
        return past_data

    @classmethod
    def add_holiday_flag(cls, df: pd.DataFrame) -> pd.DataFrame:
        """Add holiday flag to dataset

        Args:
            df (pd.DataFrame): dataset without holiday flag

        Returns:
            pd.DataFrame: dataset with holiday flag
        """
        holiday = df[['datetime']].drop_duplicates()
        holiday['tomorrow'] = holiday['datetime'] + pd.to_timedelta(1, 'd')
        holiday['tomorrow_weekday'] = holiday['tomorrow'].dt.weekday
        holiday['is_tomorrow_holiday'] = holiday['tomorrow'].map(jpholiday.is_holiday)
        holiday['holiday_flag'] = holiday.eval(
            "tomorrow_weekday in (5, 6) or is_tomorrow_holiday"
        )

        return pd.merge(df, holiday[['datetime', 'tomorrow_weekday', 'holiday_flag']], on=['datetime'], how='inner')

    @classmethod
    def create_features(cls, df: pd.DataFrame) -> pd.DataFrame:
        """Create features for prediction

        Args:
            df (pd.DataFrame): input raw data

        Returns:
            pd.DataFrame: processed data
        """

        # expand date/time
        df['month'] = df['datetime'].dt.month
        df['day'] = df['datetime'].dt.day
        df['hour'] = df['datetime'].dt.hour

        df = cls.add_holiday_flag(df)
        df['section'] = df['start_code'].astype(str) + '-' + df['end_code'].astype(str)

        # section length (km)
        df['direction_code'] = df['direction'].map({'上り':-1, '下り':1})
        df['section_length'] = df.eval('direction_code * (end_KP - start_KP)')

        past_data = cls.load_past_data(df)
        for i in range(1, consts.use_past_days+1):
            past_df = past_data[i]
            df = pd.merge(df, past_df, on=['hour'] + KEY, how='inner', suffixes=['', f'_{i}days_before'])
        return df

    @classmethod
    def save_and_cleanup_past_data(cls, df: pd.DataFrame, dt: datetime.datetime):
        """Save input data and cleanup old data

        Args:
            dt (datetime.datetime): datetime of input data
        """

        past_df = df.copy()
        past_df['hour'] = df['datetime'].dt.hour
        df[['hour'] + KEY + consts.past_cols].to_pickle(f'past_data{dt.strftime("%Y-%m-%d")}.pkl')
        dt_to_remove = dt - datetime.timedelta(days=consts.use_past_days + 1)
        if os.path.exists(f'past_data{dt_to_remove.strftime("%Y-%m-%d")}.pkl'):
            os.remove(f'past_data{dt_to_remove.strftime("%Y-%m-%d")}.pkl')

    @classmethod
    def predict(cls, input):
        """Predict method

        Args:
            input: meta data of the sample you want to make inference from (DataFrame)

        Returns:
            prediction: Inference for the given input. Return columns must be KEY(DataFrame).

        Tips:
            You can use past data by writing "cls.data".
        """

        input['datetime'] = pd.to_datetime(input['datetime'], format='%Y-%m-%d')
        cls.today = input['datetime'].iloc[0]
        input = cls.create_features(input)
        input = input.astype({ col:'category' for col in consts.categorical_features })

        for k in range(1, consts.K+1):
            # features from traffic counter
            tc_data = pd.read_csv(f'TCstats_fold{k}of{consts.K}.csv')
            model_input = pd.merge(input, tc_data, on=['start_code', 'end_code', 'holiday_flag', 'hour'], how='left')

            input[f'pred_speed_diff_{k}'] = cls.models[k].predict(model_input[consts.features])

        input['pred_speed_diff'] = input.filter(like='pred_speed_diff_').min(axis=1)
        input['prediction'] = input.eval("speed + pred_speed_diff < 40.0")
        prediction = input[['datetime', 'start_code', 'end_code', 'prediction']].copy()
        prediction = prediction.astype({
            'datetime': "datetime64[ns]",
            'start_code': int,
            'end_code': int,
            'prediction': int,
        })
        cls.save_and_cleanup_past_data(input, cls.today)
        return prediction