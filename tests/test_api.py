import pytest
from fastapi.testclient import TestClient

from src.api.main import app


VALID_PAYLOAD = {
    "tenure": 12,
    "MonthlyCharges": 70.35,
    "TotalCharges": 844.20,
    "Partner": "Yes",
    "Dependents": "No",
    "PhoneService": "Yes",
    "PaperlessBilling": "Yes",
    "gender": "Female",
    "SeniorCitizen": 0,
    "OnlineSecurity": "No",
    "OnlineBackup": "Yes",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "Yes",
    "StreamingMovies": "No",
    "Contract": "Month-to-month",
    "InternetService": "Fiber optic",
    "PaymentMethod": "Electronic check",
    "MultipleLines": "No",
}


@pytest.fixture
def client():
    # TestClient как контекстный менеджер запускает lifespan —
    # реальная модель грузится один раз на все тесты в фикстуре.
    with TestClient(app) as test_client:
        yield test_client


def test_predict_valid_payload_returns_200(client):
    response = client.post("/predict", json=VALID_PAYLOAD)
    assert response.status_code == 200


def test_predict_response_shape(client):
    response = client.post("/predict", json=VALID_PAYLOAD)
    body = response.json()
    assert set(body.keys()) == {"churn_probability", "will_churn", "threshold_used"}
    assert isinstance(body["will_churn"], bool)
    assert isinstance(body["churn_probability"], float)


def test_predict_probability_in_valid_range(client):
    response = client.post("/predict", json=VALID_PAYLOAD)
    probability = response.json()["churn_probability"]
    assert 0.0 <= probability <= 1.0


def test_predict_decision_matches_threshold(client):
    response = client.post("/predict", json=VALID_PAYLOAD)
    body = response.json()
    expected = body["churn_probability"] >= body["threshold_used"]
    assert body["will_churn"] == expected


def test_predict_missing_required_field_returns_422(client):
    payload = VALID_PAYLOAD.copy()
    del payload["tenure"]
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_predict_unknown_enum_value_returns_422(client):
    payload = VALID_PAYLOAD.copy()
    payload["Contract"] = "Three years"  # значения нет среди допустимых
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_predict_negative_tenure_returns_422(client):
    payload = VALID_PAYLOAD.copy()
    payload["tenure"] = -1
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_predict_inconsistent_internet_fields_returns_422(client):
    """Кросс-полевой валидатор: InternetService=No, но OnlineSecurity=Yes."""
    payload = VALID_PAYLOAD.copy()
    payload["InternetService"] = "No"
    payload["OnlineSecurity"] = "Yes"
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_predict_consistent_no_internet_service_is_valid(client):
    """Полностью согласованный набор при InternetService=No должен проходить."""
    payload = VALID_PAYLOAD.copy()
    payload["InternetService"] = "No"
    dependent_fields = [
        "OnlineSecurity", "OnlineBackup", "DeviceProtection",
        "TechSupport", "StreamingTV", "StreamingMovies",
    ]
    for field in dependent_fields:
        payload[field] = "No"
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
