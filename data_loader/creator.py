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

    # Align requested features with the actual dataset columns in a
    # case-insensitive fashion.  The cached BitMEX fixture stores OHLCV
    # fields in lower-case whereas some configuration files still reference
    # the historic capitalised names.  Resolving the mapping here keeps the
    # configuration backward compatible while maintaining flexibility for
    # future datasets.
    columns_lookup = {col.lower(): col for col in dataset.columns}
    resolved_features = []
    missing_features = []
    for feature in features:
        key = feature.lower()
        if key in columns_lookup:
            resolved_features.append(columns_lookup[key])
        else:
            missing_features.append(feature)

    if missing_features:
        raise KeyError(f"Requested features {missing_features} not found in dataset columns {list(dataset.columns)}")

    features = resolved_features

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
    
    # --- [بداية الكود المعدل] ---
    # 1. حدد الميزات أولاً (بدون التاريخ)
    #    نحن نستبعد عمود التاريخ من قائمة الميزات قبل تحويله إلى مصفوفة
    features = [f for f in df1.columns if f.lower() != 'date']

    # 2. أنشئ المصفوفة "arr" بناءً على هذه الميزات (الرقمية فقط)
    arr = np.array(df1[features])
    # --- [نهاية الكود المعدل] ---

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

    # "features" هنا يجب أن تكون القائمة الرقمية التي أنشأناها في الخطوة 1
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
    def _match_column(columns, target):
        target_lower = target.lower()
        for column in columns:
            if column.lower() == target_lower:
                return column
        return None

    last_col = []
    for name in features:
        last_col.append(f'{name}_day{counter_date-1}')
    last_col.append('prediction')

    # Remove the columns that should remain in the prediction frame using the
    # actual column keys to avoid case-mismatch issues when cached datasets use
    # lower-case labels.
    for base in ('High', 'Low', 'Mean'):
        column_name = _match_column(last_col, f'{base}_day{counter_date-1}')
        if column_name is not None:
            last_col.remove(column_name)

    drop_targets = []
    for col in last_col:
        match = _match_column(data_frame.columns, col)
        if match is not None:
            drop_targets.append(match)
    if drop_targets:
        data_frame.drop(drop_targets, axis=1, inplace=True)

    profit_frame = data_frame.copy()

    rename_prediction = {}
    for source, target in (
        (f'High_day{counter_date-1}', 'predicted_high'),
        (f'Low_day{counter_date-1}', 'predicted_low'),
        (f'Mean_day{counter_date-1}', 'prediction'),
    ):
        column_name = _match_column(data_frame.columns, source)
        if column_name is not None:
            rename_prediction[column_name] = target

    if rename_prediction:
        data_frame = data_frame.rename(rename_prediction, axis=1)

    profit_columns = ['Date']
    for base in ('Low', 'High', 'Close', 'Open', 'Volume'):
        column_name = _match_column(profit_frame.columns, f'{base}_day{counter_date-1}')
        if column_name is None:
            column_name = _match_column(profit_frame.columns, f'{base.lower()}_day{counter_date-1}')
        if column_name is not None:
            profit_columns.append(column_name)

    profit_calculator = profit_frame[profit_columns]

    rename_profit = {}
    for base in ('High', 'Low', 'Open', 'Close', 'Volume'):
        column_name = _match_column(profit_calculator.columns, f'{base}_day{counter_date - 1}')
        if column_name is not None:
            rename_profit[column_name] = base

    if rename_profit:
        profit_calculator = profit_calculator.rename(rename_profit, axis=1)

    return data_frame, profit_calculator
