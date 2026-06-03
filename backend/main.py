from __future__ import annotations

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from ml_models import predict_digit, predict_ufo


app = FastAPI(
    title="Session 406 ML Web App API",
    description="FastAPI backend for MNIST-style digit recognition and UFO country prediction.",
)


class DigitRequest(BaseModel):
    pixels: list[float] = Field(..., description="Flattened 8x8 or 28x28 grayscale pixels.")


class UfoRequest(BaseModel):
    seconds: float = Field(..., ge=1, le=120)
    latitude: float = Field(..., ge=-90, le=90)
    longitude: float = Field(..., ge=-180, le=180)


@app.get("/")
def root() -> dict[str, str]:
    return {"message": "Session 406 ML API is running."}


@app.post("/predict-digit")
def digit_endpoint(payload: DigitRequest) -> dict:
    try:
        prediction = predict_digit(payload.pixels)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return prediction.__dict__


@app.post("/predict-ufo")
def ufo_endpoint(payload: UfoRequest) -> dict:
    prediction = predict_ufo(payload.seconds, payload.latitude, payload.longitude)
    return prediction.__dict__

