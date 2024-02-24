import pandas as pd
import numpy as np
import pickle
import datetime
import jpholiday
from sklearn.preprocessing import LabelEncoder
import pickle
from sklearn.multioutput import ClassifierChain

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
        
        cls.n_splits = 6
        cls.model = {}
        for n in range(cls.n_splits):
            cls.model[n] = pickle.load(open(f'../model/lgb_fold{n}.pickle', 'rb'))

        cls.data = inference_df

        return True

    @classmethod
    def predict(cls, input):
        """Predict method

        Args:
            input: meta data of the sample you want to make inference from (DataFrame)

        Returns:
            prediction: Inference for the given input. Return columns must be ['datetime', 'start_code', 'end_code'](DataFrame).
        
        Tips:
            You can use past data by writing "cls.data".
        """

        cat_cols = ["start_code", "end_code", "start_pref_code", "end_pref_code", "road_code", "dayofweek", "holiday", "direction", "month"] #9
        num_cols = ["day", "search_unspec_1d", "start_lat", "end_lat", "start_lng", "end_lng",
            "start_degree", "end_degree", "KP", "limit_speed", "start_KP", "end_KP",
           "holiday_before", "holiday_after"] #12
        num_1h_cols = ["OCC", "allCars", "speed", "search_1h",] #3
        rem_cols = ["datetime", "year", "hour"] #3
        tar_col = "is_congestion" #1
        key_cols = ["date", "section"] #2

        def expand_datetime(df):
            if 'datetime' in df.columns:
                df['date'] = pd.to_datetime(df['datetime'].dt.date)
                df['year'] = df['datetime'].dt.year
                df['month'] = df['datetime'].dt.month
                df['day'] = df['datetime'].dt.day
                df['hour'] = df['datetime'].dt.hour
                # print(df.info())
            # if 'date' in df.columns:
            #     df['year'] = df['date'].dt.year
            #     df['month'] = df['date'].dt.month
            #     df['day'] = df['date'].dt.day

            return df

        def extract_dataset(train_df):
            train_df = expand_datetime(train_df)
            train_df['dayofweek'] = train_df['datetime'].dt.weekday
            train_df['section'] = train_df['start_code'].astype(str)+'_'+train_df['end_code'].astype(str)
            # train_df = train_df[feature_cols]

            return train_df

        def holiday_add_a2(df, date_col="date"):
            merged_t = df.copy()
            
            # df_holiday = pd.read_csv("./calendar.csv")
            df_holiday = pd.read_pickle("./calendar.pickle")
            # df_holiday[date_col] = pd.to_datetime(df_holiday[date_col])
            
            # 結合
            # display(df_holiday[date_col].value_counts())
            df_holiday[date_col] = pd.to_datetime(df_holiday[date_col]) # 型の調整
            merged_t[date_col] = pd.to_datetime(merged_t[date_col]) # 型の調整
            merged_t = pd.merge(merged_t, df_holiday[[date_col, "holiday", "holiday_before", "holiday_after"]], on=date_col, how="left")
            
            return merged_t

        
        def convert_df_for_rnn(df, cat_cols, num_cols, num_1h_cols, rem_cols, tar_col, key_cols):
    
            """
            時間固有：occ, allcars, speed, is_con(traget), hour, search_1h
            """
            
            # 例外処理
            df["search_1h"] = df["search_1h"].fillna(-1)
            df["search_unspec_1d"] = df["search_unspec_1d"].fillna(-1)
                
            # 日にち単位にまとめる
            df_day = df[df["hour"]==10] # 適当な時間
            df_day = df_day[key_cols+cat_cols+num_cols]
            
            # object化
            for col in cat_cols:
                df_day[col] = df_day[col].astype("category")
            
            # 時間に依存する項目のみ
            df_1h = df.pivot_table(index=key_cols, columns=["hour"], values=num_1h_cols+[tar_col], aggfunc="mean")
            col_news = [str(col[0])+"_"+str(col[1]) for col in df_1h.columns]
            df_1h.columns = col_news
            df_1h = df_1h.reset_index()
            
            # merge
            df_day = pd.merge(df_day, df_1h, on=key_cols, how="left")
            
            # one hot
            # df_day = pd.get_dummies(df_day, drop_first=True, columns=cat_cols)
            
            # le
            with open('../src/features/le_dict.pkl', 'rb') as web:
                le = pickle.load(web)
            for c in cat_cols:
                df_day[c] = le[c].transform(df_day[c])
            
            # # sort
            # df_day = df_day.sort_values(key_cols).reset_index(drop=True)
            
            feature_cols = []
            tar_cols = []
            for col in df_day.columns:
                if (tar_col in col)|(col in key_cols):
                    if tar_col in col:
                        tar_cols.append(col)
                else:
                    feature_cols.append(col)
                    
            # 目的変数を前日にする&当日の実績を変数に設定
            df_yesterday = df_day.copy(deep=True)
            df_yesterday["date"] = df_yesterday["date"] - datetime.timedelta(days=1)
            # df_day = df_day.drop(tar_cols, axis=1) ##ここを変更する
            old_tra_cols = [i.replace("is_congestion", "old") for i in tar_cols]
            feature_cols += old_tra_cols
            df_day = df_day.rename(dict(zip(tar_cols, old_tra_cols)), axis=1)
            # df_yesterday = df_yesterday[key_cols+tar_cols]
            # df_day = pd.merge(df_day, df_yesterday, on=key_cols, how="left")
            # df_day = df_day.dropna(subset=tar_cols)
            
            return df_day, feature_cols, tar_cols

        # print(input.info())
        train = input.copy()
        train = extract_dataset(train)
        train = holiday_add_a2(train)
        # print(train.shape)
        df, feature_cols, tar_cols = convert_df_for_rnn(train, cat_cols, num_cols, num_1h_cols, rem_cols, tar_col, key_cols)
        # print(df.shape)
        # for col in df.columns:
        #     print(col)

        pred = np.zeros((len(df), 24))
        for n in range(cls.n_splits):
            pred += cls.model[n].predict_proba(df[feature_cols])/cls.n_splits
        pred = (pred>0.5)*1
        # print(df.shape)
        # print(pred.shape)
        pred = pred.ravel()
        # print(pred.shape)
        train["prediction"] = pred
        prediction = train[['datetime', 'start_code', 'end_code', 'prediction']]

        return prediction