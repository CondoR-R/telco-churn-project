FROM python:3.13-slim

WORKDIR /app

COPY requirements.txt ./

RUN pip install -r requirements.txt

COPY models/ ./models/
COPY src/ ./src/

EXPOSE 8000

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
