# CIFAR-10 Image Classification Web App

This project demonstrates a simple web application for classifying images using a pre-trained Convolutional Neural Network (CNN) model trained on the CIFAR-10 dataset. Users can upload an image, and the web app will predict its class among 10 categories.

## How it Works

- **Model**: The web application uses a pre-trained CNN model trained on the CIFAR-10 dataset. The model is trained to classify images into one of the following classes: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, or truck.

- **Web Interface**: The application provides a simple web interface where users can upload an image.

- **Prediction**: When the user uploads an image, the application preprocesses the image and feeds it into the pre-trained model. The model predicts the class probabilities for each category.

- **Result**: The predicted class along with confidence scores is displayed to the user on a result page.

## Installation

1. Clone the repository:

    ```bash
    git clone https://github.com/yourusername/cifar10-flask-app.git
    cd cifar10-flask-app
    ```

2. Install the required dependencies:

    ```bash
    pip install -r requirements.txt
    ```

3. Download the pre-trained model (`cifar10_cnn_model.h5`) and place it in the project directory.

4. Run the Flask app:

    ```bash
    python app.py
    ```

5. Open your web browser and go to `http://127.0.0.1:5000/` to access the web application.


## Technologies Used

- Python
- Flask
- TensorFlow
- HTML/CSS

## Acknowledgments

- The pre-trained CNN model used in this project is trained on the CIFAR-10 dataset.
- CIFAR-10 dataset: https://www.cs.toronto.edu/~kriz/cifar.html
