import numpy as np
import pandas as pd
from datetime import datetime
from .indicators import calculate_indicators, add_indicators_to_dataset


def preprocess(dataset, cfg, logger=None):
    # فلترة البيانات حسب التاريخ
    dataset = dataset[
        (dataset['Date'] > cfg.dataset_loader.train_start_date) & (dataset['Date'] < cfg.dataset_loader.valid_end_date)
    ]

    # تحديد الأعمدة المستخدمة
    if cfg.dataset_loader.features is not None:
        features = cfg.dataset_loader.features.split(',')
        features = [s.strip() for s in features]
    else:
        features = dataset.columns.tolist()
        if 'date' in features:
            features.remove('date')

    # استخراج العمود الخاص بالتاريخ
    dates = dataset['Date']
    df = dataset[features]

    # توحيد أسماء الأعمدة
    if 'low' in df.columns:
        df = df.rename({'low': 'Low'}, axis=1)
    if 'high' in df.columns:
        df = df.rename({'high': 'High'}, axis=1)
    if 'open' in df.columns:
        df = df.rename({'open': 'Open'}, axis=1)
    if 'close' in df.columns:
        df = df.rename({'close': 'Close'}, axis=1)
    if 'volume' in df.columns:
        df = df.rename({'volume': 'Volume'}, axis=1)

    # حساب المتوسط
    try:
        df['Mean'] = (df['Low'] + df['High']) / 2
    except Exception:
        if logger is not None:
            logger.error('Your dataset_loader should have High and Low columns')

    # تنظيف البيانات
    df = df.dropna()
    df1 = df.drop('Mean', axis=1)
    arr = np.array(df1)

    # حساب المؤشرات الفنية
    indicators = calculate_indicators(
        mean_=np.array(df.Mean),
        low_=np.array(df.Low),
        high_=np.array(df.High),
        open_=np.array(df.Open),
        close_=np.array(df.Close),
        volume_=np.array(df.Volume)
    )

    indicators_names = list(cfg.dataset_loader.indicators_names.split(' '))
    arr1, dates = add_indicators_to_dataset(indicators, indicators_names, dates, mean_=np.array(df.Mean))
    arr = np.concatenate((arr[100:], arr1), axis=1)

    features = [f for f in features if f != 'date']
    features = features + indicators_names
    dataset, profit_calculator = create_dataset(
        arr,
        list(dates),
        look_back=cfg.dataset_loader.window_size,
        features=features,
        prediction_window=getattr(cfg.dataset_loader, 'prediction_window', 1)
    )
    return dataset, profit_calculator


def create_dataset(dataset, dates, look_back, features, prediction_window=1):
    data_x = []
    for i in range(len(dataset) - look_back - prediction_window):
        a = dataset[i:(i + look_back), :]
        a = a.reshape(-1)
        d = datetime.strptime(str(dates[i]).split('+')[0].split('.')[0], '%Y-%m-%d %H:%M:%S')
        b = [d]
        b = b + a.tolist()
        target_index = i + look_back + prediction_window - 1
        b.append(dataset[target_index, :][-1])
        data_x.append(b)

    data_x = np.array(data_x)

    cols = ['Date']
    counter = 0
    counter_date = 0
    for i in range(data_x.shape[1] - 2):
        name = features[counter]
        cols.append(f'{name}_day{counter_date}')
        counter += 1
        if counter >= len(features):
            counter = 0
            counter_date += 1

    cols.append('prediction')

    data_frame = pd.DataFrame(data_x, columns=cols)
    last_col = []
    for i in range(len(features)):
        name = features[i]
        last_col.append(f'{name}_day{counter_date-1}')
    last_col.append('prediction')
    last_col.remove(f'High_day{counter_date-1}')
    last_col.remove(f'Low_day{counter_date - 1}')
    last_col.remove(f'mean_day{counter_date - 1}')

    profit_calculator = data_frame.copy()[[
        'Date',
        f'Low_day{counter_date-1}', f'High_day{counter_date-1}',
        f'close_day{counter_date-1}', f'open_day{counter_date-1}',
        f'volume_day{counter_date-1}'
    ]]

    data_frame.drop(last_col, axis=1, inplace=True)
    data_frame = data_frame.rename({
        f'High_day{counter_date-1}': 'predicted_high',
        f'Low_day{counter_date-1}': 'predicted_low',
        f'mean_day{counter_date-1}': 'prediction'
    }, axis=1)

    profit_calculator = profit_calculator.rename({
        f'High_day{counter_date - 1}': 'High',
        f'Low_day{counter_date - 1}': 'Low',
        f'open_day{counter_date - 1}': 'Open',
        f'close_day{counter_date - 1}': 'Close',
        f'volume_day{counter_date - 1}': 'Volume'
    }, axis=1)

    return data_frame, profit_calculator
