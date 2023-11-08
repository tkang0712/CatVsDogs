
import streamlit as st
from fastai.vision.all import *
st.write("Cats vs. Dogs Classifier")
st.text("Builtport stream by Theo Kang")

def label_func(f): return f[0].isupper()

model = load_learner('my_model.pkl')

def predict(image):
    img = PILImage.create(image)
    pred_class, pred_idx, outputs = model.predict(img)
    likelihood_is_cat = outputs[1].item()
    if likelihood_is_cat == 1:
        return "Cat"
    elif likelihood_is_cat < 0.01:
        return "Dog"
    else:
        return "Not sure... try another picture!"
st.title("Cats vs. Dogs Classifier")
st.write("Upload an image, and I'll tell you whether it's a cat or dog.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

    if st.button("Predict"):
        prediction = predict(uploaded_file)
        st.write(prediction)

st.text("Built with Streamlit and Fastai")
