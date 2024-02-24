use_past_days = 0
past_cols = ['OCC', 'allCars', 'speed', 'is_congestion', 'search_1h', 'search_unspec_1d', 'tomorrow_weekday']
numerical_features = [
    'month',
    'day',
    'hour',
    'allCars',
    'OCC',
    'speed',
    'cars_mean',
    'cars_median',
    'cars_max',
    'cars_min',
    'occ_mean',
    'occ_median',
    'occ_max',
    'occ_min',
    'speed_mean',
    'speed_min',
    'speed_median',
    'speed_max',
    'start_lat',
    'end_lat',
    'start_lng',
    'end_lng',
    'start_degree',
    'end_degree',
    'KP',
    'start_KP',
    'end_KP',
    'section_length',
    'limit_speed',
    'search_1h',
    'search_unspec_1d',
    'holiday_flag',
    'congestion_rate',
    'tomorrow_weekday'
]
numerical_features += [f'{col}_{i}days_before' 
                       for i in range(1, use_past_days+1) 
                       for col in past_cols]
categorical_features = [
    'section',
    # 'start_pref_code',
    # 'end_pref_code',
    'direction',
    'road_code'
]
features = numerical_features + categorical_features
K = 4