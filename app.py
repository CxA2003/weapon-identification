import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# Replace with model path
def load_model():
    model = tf.keras.models.load_model('weapon_class.h5')
    return model

model = load_model()

st.title('Is there a weapon?')
st.write('Streamlit app made by Cesar Fernandez')

page = st.sidebar.selectbox(
    'Page',
    ('Home', 'Weapon Identifier')
)

base_css = """
    <style>
    .st-emotion-cache-h4xjwg {
        background-color: #060606;
        color: #ECECEC;
    }
    .st-emotion-cache-6qob1r {
        background-color: #060606;
    }
    </style>
    """

st.markdown(base_css, unsafe_allow_html=True)

if page == 'Home':
    css_code = """
    <style>
    h1, h3 {
        color: white
    }
    header {
        background-color: #1e1e1e;
        color: white;
    }
    .main {
        background-color: #3e4172;
        opacity: 1;
        background: radial-gradient(circle, transparent 20%, #3e4172 20%, #3e4172 80%, transparent 80%, transparent), radial-gradient(circle, transparent 20%, #3e4172 20%, #3e4172 80%, transparent 80%, transparent) 10px 10px, linear-gradient(#000000 0.8px, transparent 0.8px) 0 -0.4px, linear-gradient(90deg, #000000 0.8px, #3e4172 0.8px) -0.4px 0;
        background-size: 20px 20px, 20px 20px, 10px 10px, 10px 10px;
    }
    .st-emotion-cache-1n76uvr {
        background-color: #1e1e1e;
        color: white;
        display: flex;
        align-items: center;
        text-align: center;
        padding: 3rem;
        border-radius: 20px;
        box-shadow: 0 0 10px #a8c0ff;
    }
    .st-emotion-cache-13ln4jf {
        width: 100%;
        padding: 6rem 1rem 6rem;
        max-width: 46rem;
    }
    .st-emotion-cache-1sno8jx img {
        border-radius: 10px;
    }
    </style>
    """

    st.markdown(css_code, unsafe_allow_html=True)
    st.subheader('About this project')
    st.write('''
This Streamlit app allows the user to upload images and determine if they contain
a weapon with a model that reached a 99% precision.
    ''')
    st.markdown("[![Foo](https://assets.bwbx.io/images/users/iqjWHBFdfxIU/iAhX.YnazclY/v1/-1x-1.jpg)]()")

if page == 'Weapon Identifier':
    css_code = """
    <style>
    h1, h3 {
        color: white
    }
    header {
        background-color: #1e1e1e;
        color: white;
    }
    .main {
        background-color: #3e4172;
        opacity: 1;
        background: radial-gradient(circle, transparent 20%, #3e4172 20%, #3e4172 80%, transparent 80%, transparent), radial-gradient(circle, transparent 20%, #3e4172 20%, #3e4172 80%, transparent 80%, transparent) 10px 10px, linear-gradient(#000000 0.8px, transparent 0.8px) 0 -0.4px, linear-gradient(90deg, #000000 0.8px, #3e4172 0.8px) -0.4px 0;
        background-size: 20px 20px, 20px 20px, 10px 10px, 10px 10px;
    }
    .st-emotion-cache-1n76uvr {
        background-color: #1e1e1e;
        color: white;
        display: flex;
        align-items: center;
        text-align: center;
        padding: 3rem;
        border-radius: 20px;
        box-shadow: 0 0 10px #a8c0ff;
    }
    .st-emotion-cache-13ln4jf {
        width: 100%;
        padding: 6rem 1rem 6rem;
        max-width: 46rem;
    }
    .st-emotion-cache-1sno8jx img {
        border-radius: 10px;
    }
    .st-emotion-cache-qgowjl p{
        color:white;
        padding-left: 1rem;
    }
    .st-emotion-cache-7oyrr6, .st-emotion-cache-9ycgxx {
        color: white;
    }
    .st-emotion-cache-1erivf3 {
        background-color: transparent;
    }
    .st-emotion-cache-15hul6a {
        background-color: #1e1e1e;
        color: white;
        border-color: #1e1e1e;
    }
    .st-emotion-cache-15hul6a::hover {
        background-color: #1e1e1e;
        color: white
    }
    .st-emotion-cache-1v0mbdj {
        margin: 0 auto;
    }
    .st-emotion-cache-7oyrr6, .st-emotion-cache-9ycgxx {
        color: white;
    }
    </style>
    """

    st.markdown(css_code, unsafe_allow_html=True)
    st.subheader("Model testing")
    st.write('Upload an image and test the model')
    uploaded_file = st.file_uploader(
        label="File uploading section",
        type=['jpg', 'jpeg', 'png'],
        accept_multiple_files=False
    )

    if uploaded_file is not None:
        # Convert the uploaded file to an image
        img = Image.open(uploaded_file)
        st.image(img, caption="Uploaded by user", width=500)

        # Preprocess the image
        img = img.resize((224, 224))
        img_array = np.array(img)
        img_array = tf.keras.applications.resnet.preprocess_input(img_array)
        img_array = np.expand_dims(img_array, axis=0)

        # Predict the result
        prediction = model.predict(img_array)
        class_label = np.argmax(prediction, axis=1)[0]
        probability = np.max(prediction)

        st.write("And the result is...")
        if class_label == 0:
            st.write(f"⚠️ Weapon detected! (Probability: {probability:.2f})")
        else:
            st.write(f"✅ No weapon detected. (Probability: {probability:.2f})")
    else:
        st.write("Please, upload an image!")
