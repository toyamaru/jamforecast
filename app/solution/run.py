import sys
import argparse
import os
import glob
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.metrics import f1_score
import warnings
warnings.simplefilter('ignore')

# モデル構築に使えるライブラリ
from submit.src.predictor import ScoringService

def make_dataset(traffic, ic_master, search_spec, search_unspec):
    # 欠損値の除外
    traffic = traffic[traffic['speed'].isnull()==False]
    ic_master.dropna(inplace=True)
    search_spec.dropna(inplace=True)
    search_unspec.dropna(inplace=True)
    
    # datetimeからdateを作成
    traffic['date'] = traffic['datetime'].apply(lambda x: x.split()[0])

    # データのマージ
    traffic = traffic.merge(ic_master, on=['start_code', 'end_code'], how='left')
    traffic = traffic.merge(search_spec, on=['datetime', 'start_code', 'end_code'], how='left')
    traffic = traffic.merge(search_unspec, on=['date', 'start_code', 'end_code'], how='left')
    traffic.sort_values(['date', 'start_code', 'end_code'], inplace=True)
    traffic.reset_index(drop=True, inplace=True)
    traffic.drop(columns='date', inplace=True)

    # データ型の変更
    traffic['datetime'] = pd.to_datetime(traffic['datetime'])

    return traffic


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exec-path', help = '/path/to/submit/src')
    parser.add_argument('--data-dir', help = '/path/to/train')
    parser.add_argument('--start-date', default = '2023-07-01', type = str, help='start date')
    parser.add_argument('--end-date', default = '2023-07-31', type = str, help='end date')
    args = parser.parse_args()

    return args

def expand_datetime(df):
    if 'datetime' in df.columns:
        df['year'] = df['datetime'].dt.year
        df['month'] = df['datetime'].dt.month
        df['day'] = df['datetime'].dt.day
        df['hour'] = df['datetime'].dt.hour
    if 'date' in df.columns:
        df['year'] = df['date'].dt.year
        df['month'] = df['date'].dt.month
        df['day'] = df['date'].dt.day
    return df

def main():
    # parse the arguments
    args = parse_args()
    exec_path = os.path.abspath(args.exec_path)
    data_dir = os.path.abspath(args.data_dir)
    start_date = args.start_date
    end_date = args.end_date
    print('\nstart date: {}, end date:{}'.format(start_date, end_date))

    # load the input data
    print('\nLoading Dataset...')
    traffic = pd.read_csv(os.path.join(data_dir, 'train.csv'))
    search_spec = pd.read_csv(os.path.join(data_dir, 'search_specified.csv'))
    search_unspec = pd.read_csv(os.path.join(data_dir, 'search_unspecified.csv'))
    ic_master = pd.read_csv(os.path.join(data_dir, 'road_local.csv'))
    log_pathes = glob.glob(f"{data_dir}/search_raw_log/*.csv")
    
    # 当日の検索数を使用できるように変更(search_spec, search_unspec)
    search_spec['datetime'] = pd.to_datetime(search_spec['datetime'])
    search_unspec['date'] = pd.to_datetime(search_unspec['date'])
    search_spec['datetime'] -= pd.to_timedelta(1, 'd')
    search_unspec['date'] -= pd.to_timedelta(1, 'd')
    search_spec['datetime'] = search_spec['datetime'].astype('str')
    search_unspec['date'] = search_unspec['date'].astype('str')

    df = make_dataset(traffic, ic_master, search_spec, search_unspec)
    train = df[df['datetime'] < start_date]
    valid = df[(df['datetime']>=start_date+' 00:00:00') & (df['datetime']<=end_date+' 23:00:00')]
    
    #学習用のログデータの抽出
    train_log_pathes = [path for path in log_pathes if path.split("/")[-1][:-4].replace("_", "-") < start_date]
    print('Done')
    
    # change the working directory
    os.chdir(exec_path)
    cwd = os.getcwd()
    print('\nMoved to {}'.format(cwd))
    model_path = os.path.join('..', 'model')
    sys.path.append(cwd)

    # load the model
    print('\nLoading the model...', end = '\r')
    model_flag = ScoringService.get_model(model_path, train, train_log_pathes)
    if model_flag:
        print('Loaded the model.')
    else:
        print('Could not load the model.')
        return None

    predictions = pd.DataFrame()
    for d, input_df in tqdm(valid.groupby(valid['datetime'].dt.date)):
        input_df = input_df.reset_index(drop=True)
        datetime_str = d.strftime('%Y/%m/%d').replace('/', '_')
        input_log = pd.read_csv(f"{data_dir}/search_raw_log/{datetime_str}.csv")
        prediction = ScoringService.predict(input_df, input_log)
        if type(prediction)!= pd.DataFrame:
            print('Invalid data type in the prediction. Must be pandas.DataFrame')
            return None 
        elif set(prediction.columns) != set(['datetime', 'start_code', 'end_code', 'KP', 'prediction']):
            print('Invalid columns name: {},  Excepted name: {}'.format(prediction.columns, {'datetime', 'start_code', 'end_code', 'prediction'}))
            return None
        predictions = pd.concat([predictions, prediction])

    results = valid[['datetime', 'start_code', 'end_code', 'KP']]
    results = pd.merge(results, predictions, on=['datetime', 'start_code', 'end_code', 'KP'], how='left')
    results['datetime'] = pd.to_datetime(results['datetime'])
    results['datetime'] += pd.to_timedelta(1, 'd')
    results = pd.merge(results, df[['datetime', 'start_code', 'end_code', 'KP', 'is_congestion']], on=['datetime', 'start_code', 'end_code', 'KP'], how='inner')
    
    #compute F1SCORE
    print('\n==================')
    print('f1_score:', round(f1_score(results['is_congestion'], results['prediction']), 4))
    print('==================')
    results.to_csv("predict.csv", index=False)


if __name__ == '__main__':
    main()