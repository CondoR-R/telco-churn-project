import pydantic
import enum
import typing

from src.constants import DEPENDED_FEATURES


class BinCatEnum(str, enum.Enum):
    yes = 'Yes'
    no = 'No'


class ChurnPredictionRequest(pydantic.BaseModel):
    # численные признаки
    tenure: int = pydantic.Field(ge=0)
    MonthlyCharges: float = pydantic.Field(ge=0)
    TotalCharges: float

    # бинарные категориальные признаки
    Partner: BinCatEnum
    Dependents: BinCatEnum
    PhoneService: BinCatEnum
    PaperlessBilling: BinCatEnum
    gender: typing.Literal['Male', 'Female']
    SeniorCitizen: typing.Literal[0, 1]

    # зависящие от InternetService признаки
    OnlineSecurity: BinCatEnum
    OnlineBackup: BinCatEnum
    DeviceProtection: BinCatEnum
    TechSupport: BinCatEnum
    StreamingTV: BinCatEnum
    StreamingMovies: BinCatEnum

    # многоклассовые категориальные признаки
    Contract: typing.Literal['Month-to-month', 'One year', 'Two year']
    InternetService: typing.Literal['DSL', 'Fiber optic', 'No']
    PaymentMethod: typing.Literal[
        'Electronic check', 
        'Mailed check', 
        'Bank transfer (automatic)', 
        'Credit card (automatic)'
    ]
    MultipleLines: typing.Literal['Yes', 'No', 'No phone service']

    # проверка на No у зависимых полей при InternetService = No
    @pydantic.model_validator(mode='after')
    def check_internet_dependent_fields(self) -> 'ChurnPredictionRequest':
        if self.InternetService == 'No':
            invalid_fields = [
                field for field in DEPENDED_FEATURES
                if getattr(self, field) != BinCatEnum.no
            ]
            if invalid_fields:
                raise ValueError(
                    f'При InternetService=No следующие поля должны быть '
                    f'"No", так как интернет-услуги недоступны: {invalid_fields}'
                )
        return self

    
class ChurnPredictionResponse(pydantic.BaseModel):
    churn_probability: float
    will_churn: bool
    threshold_used: float
