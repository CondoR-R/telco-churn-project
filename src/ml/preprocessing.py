import pandas as pd

DEPENDED_FEATURES = [
    'OnlineSecurity', 
    'OnlineBackup', 
    'DeviceProtection', 
    'TechSupport', 
    'StreamingTV', 
    'StreamingMovies'
]

class InvalidClientDataError(ValueError):
    pass

def processing_TotalCharges(raw_df: pd.DataFrame) -> pd.DataFrame:
    '''
    Принимает сырой DataFrame и возвращает DataFrame с TotalCharges 
    приведённым к числовому типу, пропуски заполнены нулями.
    Пробрасывает исключение в случае, если у объекта tenure != 0, 
    а TotalCharges пустой.
    '''
    res_df = raw_df.copy()
    res_df['TotalCharges_num'] = pd.to_numeric(raw_df['TotalCharges'], errors='coerce')

    is_invalid_data = (res_df['TotalCharges_num'].isna()) & (res_df['tenure'] != 0)
    if is_invalid_data.any():
        raise InvalidClientDataError('При tenure != 0 TotalCharges не может быть пустым')

    res_df['TotalCharges_num'] = res_df['TotalCharges_num'].fillna(0)
    res_df = res_df.drop('TotalCharges', axis=1)

    return res_df


def features_to_bin(raw_df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    '''
    Приводит категориальные признаки с 0/1. Принимает DataFrame и 
    список признаков, которые необходимо привести к 0/1, возвращает 
    новый DataFrame с бинарными категориальными признаками. Старые столбцы
    категориальных призанков удаляет.
    '''
    res_df = raw_df.copy()
    for feature in features:
        res_df[f'{feature}_bin'] = (res_df[feature] == 'Yes').astype('uint8')
        res_df = res_df.drop(feature, axis=1)
    return res_df


def depended_features_to_bin(raw_df: pd.DataFrame) -> pd.DataFrame:
    '''
    Принимает DataFrame после функции processing_TotalCharges, для 
    списка признаков, зависящих от InternetService, схлопывает значения
    "No internet service" и "No" в "No". Возвращает DataFrame с 
    получившимися бинарными признаками.
    '''
    return features_to_bin(raw_df, DEPENDED_FEATURES)


def encoding_bin_cat_features(raw_df: pd.DataFrame) -> pd.DataFrame:
    '''
    Приводит Partner, Dependents, PhoneService, PaperlessBilling к 0/1.
    '''
    features = ['Partner', 'Dependents', 'PaperlessBilling']
    return features_to_bin(raw_df, features)


def encoding_poly_cat_features(raw_df: pd.DataFrame) -> pd.DataFrame:
    '''
    Кодирует многозначные категориальные признаки. Contract, 
    InternetService как one-hot, PaymentMethod превращает в 
    бинарный категориальный признак Electronic_check.
    '''
    def encoding_feature(df: pd.DataFrame, feature: str, values: list[str]):
        feature_cols = pd.DataFrame()
        for value in values:
            feature_cols[f'{feature}_{value}'] = df[feature] == value
        df = pd.concat([df, feature_cols.astype('uint8')], axis=1)
        return df
    
    res_df = raw_df.copy()

    contract_values = ['Month-to-month', 'One year', 'Two year']
    internet_service_values = ['DSL', 'Fiber optic', 'No']

    res_df = encoding_feature(res_df, 'Contract', contract_values)
    res_df = encoding_feature(res_df, 'InternetService', internet_service_values)

    res_df['Electronic_check'] = (res_df['PaymentMethod'] == 'Electronic check').astype('uint8')

    res_df = res_df.drop(['Contract', 'InternetService', 'PaymentMethod'], axis=1)

    return res_df


def delete_features(raw_df: pd.DataFrame) -> pd.DataFrame:
    '''
    Удаляет признаки, не учавствующие в предсказании (gender, 
    PhoneService, MultipleLines и customerID)
    '''
    return raw_df.drop(['PhoneService', 'MultipleLines', 'gender', 'customerID'], axis=1)


def alignment_cols(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    '''
    Принимает DataFrame после preprocessing и список ожидаемых 
    колонок в нужном порядке и: 
    1. переупорядочивает колонки строго в порядке, в котором 
    их видел pipeline при обучении;
    2. кидает ошибку, если какая-то ожидаемая колонка не 
    может быть восстановлена (например, пришло совсем незнакомое 
    значение категориального признака).
    '''
    missing = set(cols) - set(df.columns)
    if missing:
        raise InvalidClientDataError(f'Отсутствуют ожидаемые колонки: {sorted(missing)}')

    return df[cols]

def prepare_features(raw_df: pd.DataFrame, expected_cols: list[str]) -> pd.DataFrame:
    '''
    Принимает сырой DataFrame и список ожидаемых моделью колонок,
    готовит данные для модели и возвращает получившийся DataFrame
    '''
    df = processing_TotalCharges(raw_df)
    df = depended_features_to_bin(df)
    df = encoding_bin_cat_features(df)
    df = encoding_poly_cat_features(df)
    df = delete_features(df)
    return alignment_cols(df, expected_cols)
