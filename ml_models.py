from __future__ import annotations

import csv
import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


_digit_model = None
_ufo_model = None
_ufo_encoder: LabelEncoder | None = None
_ufo_accuracy: float | None = None
_MODEL_DIR = Path(__file__).resolve().parent / "backend"
_DIGIT_MODEL_PATH = _MODEL_DIR / "mnist_cnn.pt"
_LEGACY_DIGIT_MODEL_PATH = _MODEL_DIR / "digit_model.joblib"
_MNIST_DATA_DIR = _MODEL_DIR / "mnist_data"
_UFO_DATA_PATH = _MODEL_DIR / "data" / "ufos.csv"
_UFO_MODEL_PATH = _MODEL_DIR / "ufo_model.pkl"
_DEVICE = torch.device("cpu")
_COUNTRY_NAMES = {
    "au": "Australia",
    "ca": "Canada",
    "de": "Germany",
    "gb": "UK",
    "us": "US",
}


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


class MnistCnn(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(32 * 7 * 7, 64),
            nn.ReLU(),
            nn.Linear(64, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


def _get_digit_model():
    global _digit_model
    if _digit_model is None:
        _remove_legacy_digit_model_cache()
        model = MnistCnn().to(_DEVICE)
        if _DIGIT_MODEL_PATH.exists():
            state_dict = torch.load(_DIGIT_MODEL_PATH, map_location=_DEVICE)
            model.load_state_dict(state_dict)
            model.eval()
            _digit_model = model
            return _digit_model

        _MODEL_DIR.mkdir(parents=True, exist_ok=True)
        _train_digit_model(model)
        torch.save(model.state_dict(), _DIGIT_MODEL_PATH)
        model.eval()
        _digit_model = model
    return _digit_model


def _remove_legacy_digit_model_cache() -> None:
    if _LEGACY_DIGIT_MODEL_PATH.exists():
        try:
            _LEGACY_DIGIT_MODEL_PATH.unlink()
        except OSError:
            pass


def _train_digit_model(model: MnistCnn, epochs: int = 3) -> None:
    torch.manual_seed(42)
    train_dataset = datasets.MNIST(
        root=str(_MNIST_DATA_DIR),
        train=True,
        download=True,
        transform=transforms.ToTensor(),
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=128,
        shuffle=True,
        num_workers=0,
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    model.train()

    for _ in range(epochs):
        for images, labels in train_loader:
            images = images.to(_DEVICE)
            labels = labels.to(_DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()


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
    tensor = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0).to(_DEVICE)
    with torch.no_grad():
        logits = model(tensor)
        probabilities = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()

    digit = int(probabilities.argmax())
    return DigitPrediction(
        digit=digit,
        confidence=float(probabilities[digit]),
        probabilities=[float(v) for v in probabilities],
    )


def _ufo_training_data() -> tuple[np.ndarray, np.ndarray]:
    rows: list[tuple[float, float, float, str]] = []
    with _UFO_DATA_PATH.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for row in reader:
            try:
                seconds = float(row["duration (seconds)"])
                latitude = float(row["latitude"])
                longitude = float(row["longitude"])
            except (KeyError, TypeError, ValueError):
                continue

            country = (row.get("country") or "").strip().lower()
            if country not in _COUNTRY_NAMES:
                continue
            if not 1 <= seconds <= 60:
                continue

            rows.append((seconds, latitude, longitude, country))

    if not rows:
        raise RuntimeError(f"No UFO training rows found in {_UFO_DATA_PATH}.")

    x = np.array([(seconds, lat, lon) for seconds, lat, lon, _ in rows], dtype=np.float64)
    y = np.array([country for _, _, _, country in rows])
    return x, y


def _get_ufo_model():
    global _ufo_model, _ufo_encoder, _ufo_accuracy
    if _ufo_model is None:
        if _UFO_MODEL_PATH.exists():
            with _UFO_MODEL_PATH.open("rb") as file:
                cached = pickle.load(file)
            _ufo_model = cached["model"]
            _ufo_encoder = cached["encoder"]
            _ufo_accuracy = float(cached["accuracy"])
            return _ufo_model, _ufo_encoder, _ufo_accuracy

        x, labels = _ufo_training_data()
        encoder = LabelEncoder()
        y = encoder.fit_transform(labels)
        x_train, x_test, y_train, y_test = train_test_split(
            x,
            y,
            test_size=0.2,
            random_state=0,
            stratify=y,
        )
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=1000, random_state=0),
        )
        model.fit(x_train, y_train)
        _ufo_model = model
        _ufo_encoder = encoder
        _ufo_accuracy = float(model.score(x_test, y_test))
        _MODEL_DIR.mkdir(parents=True, exist_ok=True)
        with _UFO_MODEL_PATH.open("wb") as file:
            pickle.dump(
                {
                    "model": _ufo_model,
                    "encoder": _ufo_encoder,
                    "accuracy": _ufo_accuracy,
                },
                file,
            )
    return _ufo_model, _ufo_encoder, _ufo_accuracy


def predict_ufo(seconds: float, latitude: float, longitude: float) -> UfoPrediction:
    model, encoder, accuracy = _get_ufo_model()
    assert encoder is not None
    features = np.array([[seconds, latitude, longitude]], dtype=float)
    probabilities = model.predict_proba(features)[0]
    class_index = int(np.argmax(probabilities))
    country_code = str(encoder.inverse_transform([class_index])[0])
    country = _COUNTRY_NAMES.get(country_code, country_code.upper())
    prob_map = {
        _COUNTRY_NAMES.get(str(label), str(label).upper()): float(probabilities[i])
        for i, label in enumerate(encoder.classes_)
    }
    return UfoPrediction(
        country=country,
        confidence=float(probabilities[class_index]),
        probabilities=prob_map,
        model_accuracy=float(accuracy or 0.0),
    )
