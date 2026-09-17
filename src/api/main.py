import fastapi
import pandas as pd

from src.api.dependencies import lifespan
from src.api.shemas import ChurnPredictionRequest, ChurnPredictionResponse
from src.ml.preprocessing import prepare_features, InvalidClientDataError

app = fastapi.FastAPI(lifespan=lifespan)

def get_artifact(request: fastapi.Request):
    return request.app.state.artifact

@app.post(
    '/predict', 
    response_model=ChurnPredictionResponse, 
    summary='Получить предсказание модели'
)
def predict(
    request: ChurnPredictionRequest,
    artifact: dict = fastapi.Depends(get_artifact),
):
    raw_df = pd.DataFrame([request.model_dump()])
    try:
        df = prepare_features(raw_df, artifact['feature_columns'])
    except InvalidClientDataError as exc:
        raise fastapi.HTTPException(status_code=422, detail=str(exc))

    churn_probability = artifact['pipeline'].predict_proba(df)[:, 1][0]
    will_churn = bool(churn_probability >= artifact['threshold'])

    return ChurnPredictionResponse(
        churn_probability=churn_probability,
        will_churn=will_churn,
        threshold_used=artifact['threshold']
    )
