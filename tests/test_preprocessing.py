import pandas as pd
import pytest

from src.ml.preprocessing import (
    InvalidClientDataError,
    processing_TotalCharges,
    depended_features_to_bin,
    encoding_bin_cat_features,
    encoding_poly_cat_features,
    delete_features,
    alignment_cols,
    prepare_features,
)


@pytest.fixture
def raw_client_row() -> pd.DataFrame:
    '''
    Один "типичный" клиент — валидный сырой объект в 
    формате исходного CSV.'''
    return pd.DataFrame([{
        'customerID': '7590-VHVEG',
        'gender': 'Female',
        'SeniorCitizen': 0,
        'Partner': 'Yes',
        'Dependents': 'No',
        'tenure': 12,
        'PhoneService': 'Yes',
        'MultipleLines': 'No',
        'InternetService': 'Fiber optic',
        'OnlineSecurity': 'No',
        'OnlineBackup': 'Yes',
        'DeviceProtection': 'No',
        'TechSupport': 'No',
        'StreamingTV': 'Yes',
        'StreamingMovies': 'No',
        'Contract': 'Month-to-month',
        'PaperlessBilling': 'Yes',
        'PaymentMethod': 'Electronic check',
        'MonthlyCharges': 70.35,
        'TotalCharges': '844.20',
    }])


# --- processing_TotalCharges ---

def test_total_charges_converts_to_numeric(raw_client_row):
    result = processing_TotalCharges(raw_client_row)
    assert result.loc[0, 'TotalCharges_num'] == pytest.approx(844.20)
    assert 'TotalCharges' not in result.columns


def test_total_charges_empty_string_with_zero_tenure_becomes_zero(raw_client_row):
    row = raw_client_row.copy()
    row.loc[0, 'tenure'] = 0
    row.loc[0, 'TotalCharges'] = ' '  # как в реальном датасете — пробел, не NaN
    result = processing_TotalCharges(row)
    assert result.loc[0, 'TotalCharges_num'] == 0


def test_total_charges_empty_with_nonzero_tenure_raises(raw_client_row):
    row = raw_client_row.copy()
    row.loc[0, 'tenure'] = 5
    row.loc[0, 'TotalCharges'] = ' '
    with pytest.raises(InvalidClientDataError):
        processing_TotalCharges(row)


# --- depended_features_to_bin ---

def test_depended_features_no_internet_service_collapses_to_zero(raw_client_row):
    row = raw_client_row.copy()
    row.loc[0, 'InternetService'] = 'No'
    row.loc[0, 'OnlineSecurity'] = 'No internet service'
    row.loc[0, 'TechSupport'] = 'No internet service'
    result = depended_features_to_bin(row)
    assert result.loc[0, 'OnlineSecurity_bin'] == 0
    assert result.loc[0, 'TechSupport_bin'] == 0
    assert 'OnlineSecurity' not in result.columns


def test_depended_features_yes_maps_to_one(raw_client_row):
    result = depended_features_to_bin(raw_client_row)
    assert result.loc[0, 'OnlineBackup_bin'] == 1
    assert result.loc[0, 'OnlineSecurity_bin'] == 0


# --- encoding_bin_cat_features ---

def test_encoding_bin_cat_features(raw_client_row):
    result = encoding_bin_cat_features(raw_client_row)
    assert result.loc[0, 'Partner_bin'] == 1
    assert result.loc[0, 'Dependents_bin'] == 0
    assert result.loc[0, 'PaperlessBilling_bin'] == 1
    for col in ['Partner', 'Dependents', 'PaperlessBilling']:
        assert col not in result.columns


# --- encoding_poly_cat_features ---

def test_encoding_poly_creates_all_onehot_columns_for_single_row(raw_client_row):
    """Ключевой тест: даже для одной строки должны появиться ВСЕ колонки
    известных категорий, а не только та, что реально встретилась."""
    result = encoding_poly_cat_features(raw_client_row)

    expected_new_cols = [
        'Contract_Month-to-month', 'Contract_One year', 'Contract_Two year',
        'InternetService_DSL', 'InternetService_Fiber optic', 'InternetService_No',
        'Electronic_check',
    ]
    for col in expected_new_cols:
        assert col in result.columns

    assert result.loc[0, 'Contract_Month-to-month'] == 1
    assert result.loc[0, 'Contract_One year'] == 0
    assert result.loc[0, 'Contract_Two year'] == 0

    assert result.loc[0, 'InternetService_Fiber optic'] == 1
    assert result.loc[0, 'InternetService_DSL'] == 0

    assert result.loc[0, 'Electronic_check'] == 1

    for col in ['Contract', 'InternetService', 'PaymentMethod']:
        assert col not in result.columns


def test_encoding_poly_electronic_check_zero_for_other_payment_method(raw_client_row):
    row = raw_client_row.copy()
    row.loc[0, 'PaymentMethod'] = 'Mailed check'
    result = encoding_poly_cat_features(row)
    assert result.loc[0, 'Electronic_check'] == 0


# --- delete_features ---

def test_delete_features_removes_expected_columns(raw_client_row):
    result = delete_features(raw_client_row)
    for col in ['PhoneService', 'MultipleLines', 'gender', 'customerID']:
        assert col not in result.columns


# --- alignment_cols ---

def test_alignment_cols_reorders_columns():
    df = pd.DataFrame({'b': [2], 'a': [1], 'c': [3]})
    result = alignment_cols(df, ['a', 'b', 'c'])
    assert list(result.columns) == ['a', 'b', 'c']


def test_alignment_cols_raises_on_missing_column():
    df = pd.DataFrame({'a': [1], 'b': [2]})
    with pytest.raises(InvalidClientDataError):
        alignment_cols(df, ['a', 'b', 'c'])


# --- prepare_features (end-to-end) ---

def test_prepare_features_end_to_end(raw_client_row):
    expected_cols = [
        'SeniorCitizen', 'tenure', 'MonthlyCharges', 'TotalCharges_num',
        'OnlineSecurity_bin', 'OnlineBackup_bin', 'DeviceProtection_bin',
        'TechSupport_bin', 'StreamingTV_bin', 'StreamingMovies_bin',
        'Partner_bin', 'Dependents_bin', 'PaperlessBilling_bin',
        'Contract_Month-to-month', 'Contract_One year', 'Contract_Two year',
        'InternetService_DSL', 'InternetService_Fiber optic', 'InternetService_No',
        'Electronic_check',
    ]
    result = prepare_features(raw_client_row, expected_cols)
    assert list(result.columns) == expected_cols
    assert len(result) == 1


def test_prepare_features_raises_on_unrecoverable_data(raw_client_row):
    row = raw_client_row.copy()
    row.loc[0, 'tenure'] = 5
    row.loc[0, 'TotalCharges'] = ' '
    with pytest.raises(InvalidClientDataError):
        prepare_features(row, expected_cols=[])
