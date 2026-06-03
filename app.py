from __future__ import annotations

import streamlit as st
from PIL import Image
from streamlit_drawable_canvas import st_canvas

from ml_models import predict_digit, predict_ufo, processed_digit_image


st.set_page_config(page_title="Session 406 ML Apps", page_icon="ML", layout="wide")

st.title("Session 406 ML Web Apps")

digit_tab, ufo_tab = st.tabs(["MNIST Digit Predictor", "UFO Country Predictor"])


with digit_tab:
    st.subheader("Draw a digit")
    left, right = st.columns([1, 1])

    with left:
        canvas = st_canvas(
            fill_color="rgba(255, 255, 255, 0)",
            stroke_width=18,
            stroke_color="#ffffff",
            background_color="#000000",
            width=280,
            height=280,
            drawing_mode="freedraw",
            key="digit_canvas",
        )

    with right:
        if canvas.image_data is None:
            st.info("Draw a digit from 0 to 9.")
        else:
            model_image = processed_digit_image(canvas.image_data)
            if model_image.max(initial=0) < 0.05:
                st.info("Draw a digit from 0 to 9.")
            else:
                result = predict_digit(canvas.image_data)
                st.metric("Prediction", result.digit, f"{result.confidence:.1%} confidence")
                st.bar_chart(
                    {
                        "probability": {
                            str(index): value
                            for index, value in enumerate(result.probabilities)
                        }
                    }
                )
                preview = Image.fromarray((model_image * 255.0).astype("uint8"))
                st.image(
                    preview.resize((140, 140), Image.Resampling.NEAREST),
                    caption="Model input: 28 x 28",
                )


with ufo_tab:
    st.subheader("Predict reported UFO country")

    col1, col2, col3 = st.columns(3)
    seconds = col1.slider("Duration in seconds", 1, 120, 30)
    latitude = col2.number_input("Latitude", min_value=-90.0, max_value=90.0, value=39.9)
    longitude = col3.number_input("Longitude", min_value=-180.0, max_value=180.0, value=-75.1)

    result = predict_ufo(seconds, latitude, longitude)
    st.metric("Predicted country", result.country.upper(), f"{result.confidence:.1%} confidence")
    st.write(f"Training accuracy: `{result.model_accuracy:.2%}`")
    st.bar_chart({"probability": result.probabilities})
