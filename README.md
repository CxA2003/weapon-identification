# Weapon Identifier

## Project Overview

The **Weapon Identifier** is a Streamlit app designed to help users determine if a gun is present in their images. The application utilizes a machine learning model to analyze the content of images and trigger an alert if a weapon is detected. This tool is intended to be used for educational and informational purposes.

## Dataset

The dataset used for this project is composed of images extracted from the [OpenImages website](https://storage.googleapis.com/openimages/web/index.html) using the [openimages Python library](https://pypi.org/project/openimages/). These images were carefully selected and categorized to train the model in distinguishing between images containing guns and those without.

## Model Details

This project employs the [VGG16](https://keras.io/api/applications/vgg/#vgg16-function) model, a well-known convolutional neural network architecture. The model has been adapted with additional layers, including [GlobalAveragePooling2D](https://www.tensorflow.org/api_docs/python/tf/keras/layers/GlobalAveragePooling2D) and [BatchNormalization](https://keras.io/api/layers/normalization_layers/batch_normalization/#batchnormalization-class), to enhance its performance in this specific task. The training process was optimized using [early stopping](https://keras.io/api/callbacks/early_stopping/#earlystopping-class) to prevent overfitting and ensure the model's generalizability.

## Installation and Usage

No specific installation is required to run this project. The entire process can be executed using web-based tools like [Kaggle Notebooks](https://www.kaggle.com/code) and [Google Colab](https://colab.research.google.com/). However, a local environment can also be used to run this program. Detailed instructions and notebooks are provided to guide users through running the application in these environments.

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

#### Steps to run on locally:

- (**STRONGLY SUGGESTED**) Create a virtual environment using the tool of your preference, whether it is [Conda](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html) or [Python Venvs](https://docs.python.org/3/library/venv.html) (For this project, Python's `.venv` will be used)
    - Command to create and activate an environment:
        - Create venv: `python -m venv /path/to/new/virtual/environment`
        - Activate venv: `source /path/to/new/virtual/environment`
    - Install all packages after activating the environment:
        - `pip install -r requirements.txt`
<<<<<<< HEAD
- Replace the content of notebooks to fit your local environment (paths are adapted to Kaggle and Google Colab)
=======
- Replace content of notebooks to fit your local environment (paths are adapted to Kaggle and Google Colab)
>>>>>>> 340cc2901908e4fab16dbb92ee5b8fe91b190036
- Once the model is exported, replace the path of your model on the `app.py` file
- Open a command line and run `python -m streamlit run app.py`, the app should run on your default web browser

## Features

- **Real-time Weapon Detection**: Upload an image, and the app will analyze it to determine if a gun is present.
- **User-Friendly Interface**: The app is built using [Streamlit](https://streamlit.io/), ensuring a smooth and intuitive user experience.

<<<<<<< HEAD
## Results

The model was tested using images from the **openimages** ([link](https://storage.googleapis.com/openimages/web/index.html)) website, the dataset is comprised of a total of 12,000 photos, 6,000 of them containing guns while the rest don't.

#### Baseline:

The baseline for predictions on this dataset is 0.5, meaning there's a 50% chance of correctly guessing if the image contains a gun or not

#### Model's performance:

The results of the model on the testing data are as follows:

|                | Predicted: Gun | Predicted: Non-Gun |
|----------------|:--------------:|:------------------:|
| **Actual: Gun**      |      1517       |        30          |
| **Actual: Non-Gun**  |      30        |        1517          |


This model displays a precision of 0.98. Which could be considered a successful outcome, compared to our baseline.

## Conclusions:

- This model successfully classifies pictures based on the presence or absence of a gun. 
- The 2% margin of error doesn't lean towards False Positives or False Negatives.
- The resulting model's file size is considerably large (~110Mb), which may cause trouble when loading into the Streamlit app (Currently working on optimizing it)

## License

This project is licensed under the MIT License. Please take a look at the [LICENSE](LICENSE) file for more details.
=======
## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.
>>>>>>> 340cc2901908e4fab16dbb92ee5b8fe91b190036
