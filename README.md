# Session 406 ML Web Apps

This repository contains two visual machine-learning web apps for Session 406:

- MNIST-style handwritten digit predictor
- UFO country predictor

The project includes:

- `app.py`: Streamlit frontend with two tabs
- `backend/main.py`: FastAPI backend for predictions
- `ml_models.py`: shared model loading and prediction utilities
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

The first launch may take longer because the app downloads MNIST, trains the CNN
once on CPU, and saves the model to `backend/mnist_cnn.pt` in the app runtime.

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
