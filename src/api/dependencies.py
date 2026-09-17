import asyncio
import contextlib
import functools

import fastapi
import joblib

from src.constants import MODEL_PATH

@contextlib.asynccontextmanager
async def lifespan(app: fastapi.FastAPI):
    loop = asyncio.get_running_loop()
    try:
        app.state.artifact = await loop.run_in_executor(
            None, functools.partial(joblib.load, MODEL_PATH)
        )
    except Exception as exc:
        raise RuntimeError(
            f'Не удалось загрузить модель из {MODEL_PATH}: {exc}'
        ) from exc
    yield

