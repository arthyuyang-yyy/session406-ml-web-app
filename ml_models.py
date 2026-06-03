from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler


_digit_model = None
_ufo_model = None
_ufo_encoder: LabelEncoder | None = None
_ufo_accuracy: float | None = None
_DEVICE = torch.device("cpu")
_MNIST_MODEL_PATH = Path(__file__).resolve().parent / "backend" / "mnist_cnn.pt"
_MNIST_DATA_DIR = Path(__file__).resolve().parent / "backend" / "mnist_data"


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
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


def _train_digit_model() -> MnistCnn:
    torch.manual_seed(42)
    _MNIST_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    train_data = datasets.MNIST(
        root=str(_MNIST_DATA_DIR),
        train=True,
        download=True,
        transform=transforms.ToTensor(),
    )
    train_loader = DataLoader(train_data, batch_size=128, shuffle=True)

    model = MnistCnn().to(_DEVICE)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    model.train()
    for _ in range(1):
        for images, labels in train_loader:
            images = images.to(_DEVICE)
            labels = labels.to(_DEVICE)

            optimizer.zero_grad()
            loss = loss_fn(model(images), labels)
            loss.backward()
            optimizer.step()

    torch.save(model.state_dict(), _MNIST_MODEL_PATH)
    model.eval()
    return model


def _get_digit_model():
    global _digit_model
    if _digit_model is None:
        model = MnistCnn().to(_DEVICE)
        if _MNIST_MODEL_PATH.exists():
            state = torch.load(_MNIST_MODEL_PATH, map_location=_DEVICE)
            model.load_state_dict(state)
            model.eval()
        else:
            model = _train_digit_model()
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


def processed_digit_image(pixels: list[float] | np.ndarray) -> np.ndarray:
    """Return the normalized 28x28 grayscale image passed to the CNN."""
    arr = np.asarray(pixels)
    if arr.ndim == 3 and arr.shape[-1] in {3, 4}:
        image = Image.fromarray(np.clip(arr, 0, 255).astype("uint8")).convert("RGBA")
        background = Image.new("RGBA", image.size, (0, 0, 0, 255))
        image = Image.alpha_composite(background, image).convert("L")
        gray = np.asarray(image, dtype=np.uint8)
        foreground = gray > 20

        if not foreground.any():
            return np.zeros((28, 28), dtype=np.float32)

        ys, xs = np.where(foreground)
        top, bottom = int(ys.min()), int(ys.max()) + 1
        left, right = int(xs.min()), int(xs.max()) + 1
        height = bottom - top
        width = right - left
        padding = max(10, int(max(height, width) * 0.25))
        top = max(0, top - padding)
        bottom = min(gray.shape[0], bottom + padding)
        left = max(0, left - padding)
        right = min(gray.shape[1], right + padding)

        image = Image.fromarray(gray[top:bottom, left:right])
        side = max(image.width, image.height)
        square = Image.new("L", (side, side), 0)
        offset = ((side - image.width) // 2, (side - image.height) // 2)
        square.paste(image, offset)
        image = square.resize((28, 28), Image.Resampling.LANCZOS)
        return np.asarray(image, dtype=np.float32) / 255.0

    arr = arr.astype(np.float32, copy=False)
    if arr.size == 784:
        return _normalize_grayscale(arr.reshape(28, 28))
    if arr.size == 64:
        scaled = _normalize_grayscale(arr.reshape(8, 8)) * 255.0
        image = Image.fromarray(
            scaled.astype("uint8")
        ).convert("L")
        image = image.resize((28, 28), Image.Resampling.LANCZOS)
        return np.asarray(image, dtype=np.float32) / 255.0

    raise ValueError("Digit input must contain RGBA canvas data, or 64 or 784 pixel values.")


def predict_digit(pixels: list[float] | np.ndarray) -> DigitPrediction:
    arr = processed_digit_image(pixels)
    model = _get_digit_model()

    features = torch.from_numpy(arr).float().unsqueeze(0).unsqueeze(0).to(_DEVICE)
    with torch.no_grad():
        logits = model(features)
        probabilities = torch.softmax(logits, dim=1).cpu().numpy()[0]

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
