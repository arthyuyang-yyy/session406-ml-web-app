from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC


_digit_model = None
_ufo_model = None
_ufo_encoder: LabelEncoder | None = None
_ufo_accuracy: float | None = None


@dataclass(frozen=True)
class DigitPrediction:
    digit: int
    confidence: float
    probabilities: list[float]


@dataclass(frozen=True)
class UfoPrediction:
    country: str
    confidence: float
    probabilities: dict[str, float]
    model_accuracy: float


def _get_digit_model():
    global _digit_model
    if _digit_model is None:
        digits = load_digits()
        x_train, _, y_train, _ = train_test_split(
            digits.data,
            digits.target,
            test_size=0.2,
            random_state=42,
            stratify=digits.target,
        )
        model = make_pipeline(
            StandardScaler(),
            SVC(gamma=0.001, C=10, probability=True, random_state=42),
        )
        model.fit(x_train, y_train)
        _digit_model = model
    return _digit_model


def processed_digit_image(pixels: list[float] | np.ndarray) -> np.ndarray:
    """Return the 8x8 grayscale image that is passed to the digit model."""
    arr = np.asarray(pixels)
    if arr.ndim == 3 and arr.shape[-1] in {3, 4}:
        image = Image.fromarray(np.clip(arr, 0, 255).astype("uint8")).convert("RGBA")
        background = Image.new("RGBA", image.size, (0, 0, 0, 255))
        image = Image.alpha_composite(background, image).convert("L")
        gray = np.asarray(image, dtype=np.uint8)
        foreground = gray > 20

        if foreground.any():
            ys, xs = np.where(foreground)
            top, bottom = int(ys.min()), int(ys.max()) + 1
            left, right = int(xs.min()), int(xs.max()) + 1
            height = bottom - top
            width = right - left
            padding = max(8, int(max(height, width) * 0.20))
            top = max(0, top - padding)
            bottom = min(gray.shape[0], bottom + padding)
            left = max(0, left - padding)
            right = min(gray.shape[1], right + padding)
            image = Image.fromarray(gray[top:bottom, left:right])

        image.thumbnail((8, 8), Image.Resampling.LANCZOS)
        centered = Image.new("L", (8, 8), 0)
        offset = ((8 - image.width) // 2, (8 - image.height) // 2)
        centered.paste(image, offset)
        return np.asarray(centered, dtype=np.float32) / 255.0 * 16.0

    arr = arr.astype(np.float32, copy=False)
    if arr.size == 784:
        image = Image.fromarray(
            np.clip(arr.reshape(28, 28), 0, 255).astype("uint8")
        )
        image = image.resize((8, 8), Image.Resampling.LANCZOS)
        arr = np.asarray(image, dtype=np.float32)
        if arr.max(initial=0) > 16:
            arr = arr / 255.0 * 16.0
        return arr
    if arr.size == 64:
        arr = arr.reshape(8, 8)
        if arr.max(initial=0) > 16:
            arr = arr / 255.0 * 16.0
        return arr.astype(np.float32, copy=False)

    raise ValueError("Digit input must contain RGBA canvas data, or 64 or 784 pixel values.")


def predict_digit(pixels: list[float] | np.ndarray) -> DigitPrediction:
    arr = processed_digit_image(pixels)
    features = arr.reshape(1, 64)

    model = _get_digit_model()
    probabilities = model.predict_proba(features)[0]
    digit = int(np.argmax(probabilities))
    return DigitPrediction(
        digit=digit,
        confidence=float(probabilities[digit]),
        probabilities=[float(v) for v in probabilities],
    )


def _ufo_training_data() -> tuple[np.ndarray, np.ndarray]:
    rows = [
        (15, 40.7, -74.0, "us"),
        (25, 34.0, -118.2, "us"),
        (50, 47.6, -122.3, "us"),
        (10, 43.7, -79.4, "ca"),
        (35, 49.3, -123.1, "ca"),
        (55, 51.0, -114.1, "ca"),
        (20, 51.5, -0.1, "gb"),
        (45, 53.5, -2.2, "gb"),
        (30, 55.9, -3.2, "gb"),
        (18, -33.9, 151.2, "au"),
        (42, -37.8, 144.9, "au"),
        (58, -27.5, 153.0, "au"),
        (12, 48.8, 2.3, "fr"),
        (38, 45.8, 4.8, "fr"),
        (52, 43.6, 1.4, "fr"),
        (16, 52.5, 13.4, "de"),
        (40, 48.1, 11.6, "de"),
        (54, 50.9, 6.9, "de"),
    ]
    x = np.array([(seconds, lat, lon) for seconds, lat, lon, _ in rows], dtype=float)
    y = np.array([country for _, _, _, country in rows])
    return x, y


def _get_ufo_model():
    global _ufo_model, _ufo_encoder, _ufo_accuracy
    if _ufo_model is None:
        x, labels = _ufo_training_data()
        encoder = LabelEncoder()
        y = encoder.fit_transform(labels)
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=1000, random_state=42),
        )
        model.fit(x, y)
        _ufo_model = model
        _ufo_encoder = encoder
        _ufo_accuracy = float(model.score(x, y))
    return _ufo_model, _ufo_encoder, _ufo_accuracy


def predict_ufo(seconds: float, latitude: float, longitude: float) -> UfoPrediction:
    model, encoder, accuracy = _get_ufo_model()
    assert encoder is not None
    features = np.array([[seconds, latitude, longitude]], dtype=float)
    probabilities = model.predict_proba(features)[0]
    class_index = int(np.argmax(probabilities))
    country = str(encoder.inverse_transform([class_index])[0])
    prob_map = {
        str(label): float(probabilities[i])
        for i, label in enumerate(encoder.classes_)
    }
    return UfoPrediction(
        country=country,
        confidence=float(probabilities[class_index]),
        probabilities=prob_map,
        model_accuracy=float(accuracy or 0.0),
    )
