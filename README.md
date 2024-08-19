# Weapon Identifier

## Project Overview

The **Weapon Identifier** is a Streamlit app designed to help users determine if a gun is present in their images. The application utilizes a machine learning model to analyze the content of images and trigger an alert if a weapon is detected. This tool is intended to be used for educational and informational purposes.

## Dataset

The dataset used for this project is composed of images extracted from the [OpenImages website](https://storage.googleapis.com/openimages/web/index.html) using the [openimages Python library](https://pypi.org/project/openimages/). These images were carefully selected and categorized to train the model in distinguishing between images containing guns and those without.

## Model Details

This project employs the [VGG16](https://keras.io/api/applications/vgg/#vgg16-function) model, a well-known convolutional neural network architecture. The model has been adapted with additional layers, including [GlobalAveragePooling2D](https://www.tensorflow.org/api_docs/python/tf/keras/layers/GlobalAveragePooling2D) and [BatchNormalization](https://keras.io/api/layers/normalization_layers/batch_normalization/#batchnormalization-class), to enhance its performance in this specific task. The training process was optimized using [early stopping](https://keras.io/api/callbacks/early_stopping/#earlystopping-class) to prevent overfitting and ensure the model's generalizability.

## Installation and Usage

No specific installation is required to run this project. The entire process can be executed using web-based tools like [Kaggle Notebooks](https://www.kaggle.com/code) and [Google Colab](https://colab.research.google.com/). Detailed instructions and notebooks are provided to guide users through running the application in these environments.

#### Steps to run on web-based tools:

- Notebook for the model:
    - Upload the **WeaponIdentifyier(Kaggle Notebooks).ipynb** file to your Kaggle Account ([tutorial here](https://www.youtube.com/watch?v=HaeoKp0akN0&t=4s))
    - Set the Accelerator to GPU T4 x2
    - Run the whole notebook
    - The resulting model should appear on the right side of the page, under the "Output" section
- Notebook for the Streamlit App:
    - Upload the **WeaponIdentifyier_streamlit_app(Colab).ipynb** file to your Google Drive, or directly to your Google Colab account ([tutorial here](https://www.youtube.com/watch?v=R3sKKvMCwTo))
    - Upload the **app.py** and **weapon_class.h5** files to your Colab session files. To achieve this, click on the folder icon, then drag and drop the files, please wait until they finish loading
    - Obtain your ngrok Authtoken:
        - Log in/Sign up on the [ngrok website](https://ngrok.com/)
        - Once on the Dashboard, head to the "Your Authtoken" section
        - Copy your personal Authtoken
    - Replace the `YOUR_AUTHTOKEN` with your own Authtoken
    - Run the whole notebook
    - The link to your running app will be displayed as the output of the last code block

## Features

- **Real-time Weapon Detection**: Upload an image, and the app will analyze it to determine if a gun is present.
- **User-Friendly Interface**: The app is built using [Streamlit](https://streamlit.io/), ensuring a smooth and intuitive user experience.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.
