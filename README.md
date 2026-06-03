# Session 406 ML Web Apps

This repository contains two visual machine-learning web apps for Session 406:

- MNIST handwritten digit predictor with 28x28 preprocessing and a small
  PyTorch CNN
- UFO country predictor trained from the course `ufos.csv` dataset

The project includes:

- `app.py`: Streamlit frontend with two tabs
- `backend/main.py`: FastAPI backend for predictions
- `ml_models.py`: shared preprocessing, model loading, and prediction utilities
- `docs/index.html`: static GitHub Pages demo

## Live Web Page

GitHub Pages serves the static visual demo from `docs/`.

Expected URL:

```text
https://arthyuyang-yyy.github.io/session406-ml-web-app/
```

## Local Streamlit App

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Open:

```text
http://localhost:8501
```

## Streamlit Community Cloud

Deploy this repository at:

```text
https://share.streamlit.io/
```

Use these settings:

- Repository: `arthyuyang-yyy/session406-ml-web-app`
- Branch: `main`
- Main file path: `app.py`
- Python version: select `3.12` in Advanced settings

The first launch may take a short moment because the digit classifier downloads
MNIST, trains a small CPU-only CNN for 3 epochs, and caches itself as
`backend/mnist_cnn.pt` in the app runtime.

The UFO predictor trains from `backend/data/ufos.csv` using the Session 406
workflow: keep sightings between 1 and 60 seconds, use seconds, latitude, and
longitude as features, and predict Australia, Canada, Germany, UK, or US. It
caches the trained model as `backend/ufo_model.pkl` in the app runtime.

## FastAPI Backend

```bash
uvicorn backend.main:app --reload
```

Open:

```text
http://localhost:8000/docs
```

## API Examples

MNIST:

```bash
curl -X POST http://localhost:8000/predict-digit \
  -H "Content-Type: application/json" \
  -d '{"pixels":[0,0,0,0,0,0,0,0]}'
```

UFO:

```bash
curl -X POST http://localhost:8000/predict-ufo \
  -H "Content-Type: application/json" \
  -d '{"seconds":20,"latitude":39.9,"longitude":-75.1}'
```
