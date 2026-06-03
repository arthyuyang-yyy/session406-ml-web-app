from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
from PIL import Image
from sklearn.datasets import load_digits
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler


_digit_model = None
_ufo_model = None
_ufo_encoder: LabelEncoder | None = None
_ufo_accuracy: float | None = None
_MODEL_DIR = Path(__file__).resolve().parent / "backend"
_DIGIT_MODEL_PATH = _MODEL_DIR / "digit_model.joblib"


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
        if _DIGIT_MODEL_PATH.exists():
            _digit_model = joblib.load(_DIGIT_MODEL_PATH)
            return _digit_model

        x, y = _digit_training_data()
        model = ExtraTreesClassifier(
            n_estimators=300,
            max_features="sqrt",
            min_samples_leaf=1,
            n_jobs=-1,
            random_state=42,
        )
        model.fit(x, y)
        _MODEL_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, _DIGIT_MODEL_PATH)
        _digit_model = model
    return _digit_model


def _normalize_grayscale(arr: np.ndarray) -> np.ndarray:
    arr = arr.astype(np.float32, copy=False)
    max_value = arr.max(initial=0)
    if max_value > 16:
        arr = arr / 255.0
    elif max_value > 1:
        arr = arr / 16.0
    return np.clip(arr, 0.0, 1.0).astype(np.float32, copy=False)


def _center_digit(image: Image.Image, output_size: int = 28, target_extent: int = 20) -> Image.Image:
    gray = image.convert("L")
    arr = np.asarray(gray, dtype=np.uint8)
    foreground = arr > 20

    if not foreground.any():
        return Image.new("L", (output_size, output_size), 0)

    ys, xs = np.where(foreground)
    top, bottom = int(ys.min()), int(ys.max()) + 1
    left, right = int(xs.min()), int(xs.max()) + 1
    cropped = gray.crop((left, top, right, bottom))

    scale = target_extent / max(cropped.width, cropped.height)
    resized_size = (
        max(1, int(round(cropped.width * scale))),
        max(1, int(round(cropped.height * scale))),
    )
    resized = cropped.resize(resized_size, Image.Resampling.LANCZOS)

    canvas = Image.new("L", (output_size, output_size), 0)
    offset = ((output_size - resized.width) // 2, (output_size - resized.height) // 2)
    canvas.paste(resized, offset)
    return canvas


def _image_to_features(image: Image.Image) -> np.ndarray:
    centered = _center_digit(image)
    return (np.asarray(centered, dtype=np.float32) / 255.0).reshape(-1)


def _digit_training_data() -> tuple[np.ndarray, np.ndarray]:
    digits = load_digits()
    features: list[np.ndarray] = []
    labels: list[int] = []
    shifts = [(-2, 0), (2, 0), (0, -2), (0, 2), (0, 0)]

    for pixels, label in zip(digits.images, digits.target):
        base = Image.fromarray((pixels / 16.0 * 255.0).astype("uint8"), mode="L")
        base = base.resize((28, 28), Image.Resampling.LANCZOS)

        variants = [base, base.rotate(-10, fillcolor=0), base.rotate(10, fillcolor=0)]
        for variant in variants:
            for dx, dy in shifts:
                shifted = Image.new("L", variant.size, 0)
                shifted.paste(variant, (dx, dy))
                features.append(_image_to_features(shifted))
                labels.append(int(label))

    return np.asarray(features, dtype=np.float32), np.asarray(labels, dtype=np.int64)


def processed_digit_image(pixels: list[float] | np.ndarray) -> np.ndarray:
    """Return the normalized 28x28 grayscale image passed to the digit model."""
    arr = np.asarray(pixels)
    if arr.ndim == 3 and arr.shape[-1] in {3, 4}:
        image = Image.fromarray(np.clip(arr, 0, 255).astype("uint8")).convert("RGBA")
        background = Image.new("RGBA", image.size, (0, 0, 0, 255))
        image = Image.alpha_composite(background, image).convert("L")
        image = _center_digit(image)
        return np.asarray(image, dtype=np.float32) / 255.0

    arr = arr.astype(np.float32, copy=False)
    if arr.size == 784:
        image = Image.fromarray((_normalize_grayscale(arr.reshape(28, 28)) * 255.0).astype("uint8"))
        image = _center_digit(image)
        return np.asarray(image, dtype=np.float32) / 255.0
    if arr.size == 64:
        scaled = _normalize_grayscale(arr.reshape(8, 8)) * 255.0
        image = Image.fromarray(scaled.astype("uint8")).convert("L")
        image = image.resize((28, 28), Image.Resampling.LANCZOS)
        image = _center_digit(image)
        return np.asarray(image, dtype=np.float32) / 255.0

    raise ValueError("Digit input must contain RGBA canvas data, or 64 or 784 pixel values.")


def predict_digit(pixels: list[float] | np.ndarray) -> DigitPrediction:
    arr = processed_digit_image(pixels)
    model = _get_digit_model()
    features = arr.reshape(1, -1)
    probabilities = model.predict_proba(features)[0]

    digit = int(probabilities.argmax())
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
