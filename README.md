# Gender-and-Age-Prediction

An application that uses deep learning to predict the age and gender of a person from their facial image. This project utilizes the UTKFace dataset and is built with TensorFlow and Keras.

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Dataset](#dataset)
- [Requirements](#requirements)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Usage](#usage)
- [Model Architecture](#model-architecture)
- [Results](#results)
- [Contributing](#contributing)


## Overview
This Age and Gender Detector uses a Convolutional Neural Network (CNN) to predict a person's age and gender based on their facial image. The model is trained on the UTKFace dataset and provides a user-friendly GUI for easy interaction.

## Features
- Age prediction in years
- Binary gender classification (Male/Female)
- Simple GUI for uploading and analyzing images
- Pre-trained model for immediate use

## Dataset
The project uses the UTKFace dataset from Kaggle:
- **Source**: [UTKFace Dataset on Kaggle](https://www.kaggle.com/datasets/jangedoo/utkface-new)
- **Description**: A large-scale face dataset with annotations of age, gender, and ethnicity
- **Format**: Images named with age, gender, and ethnicity labels (e.g., [age]_[gender]_[ethnicity]_[date&time].jpg)

## Requirements
- Python 3.6+
- TensorFlow 2.x
- OpenCV
- NumPy
- Matplotlib
- scikit-learn
- PIL (Pillow)
- tkinter

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/kurakula-prashanth/Gender-and-Age-Prediction.git
   cd age-gender-detector
   ```

2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

3. Download the UTKFace dataset from [Kaggle](https://www.kaggle.com/datasets/jangedoo/utkface-new) and extract it to the project directory in a folder named `UTkFace`.

## Project Structure
```
age-gender-detector/
├── data_preparation.py       # Script to preprocess and save the dataset
├── model_training.py         # Script to train the CNN model
├── gui_application.py        # GUI interface for predictions
├── Age_Sex_Detection.keras   # Pre-trained model
├── README.md                 # Project documentation
└── UTkFace/                  # Dataset directory
    ├── image.npy             # Preprocessed images
    ├── ages.npy              # Age labels
    └── genders.npy           # Gender labels
```

## Usage
1. Prepare the dataset:
   ```bash
   python data_preparation.py
   ```

2. Train the model (optional, pre-trained model is provided):
   ```bash
   python model_training.py
   ```

3. Run the GUI application:
   ```bash
   python gui_application.py
   ```

4. In the GUI:
   - Click "Upload an Image" to select an image
   - Click "Detect Image" to get predictions

## Model Architecture
The model uses a CNN architecture with the following components:
- Four convolutional blocks (each with Conv2D, Dropout, Activation, and MaxPooling)
- Flatten layer to convert feature maps to vector
- Two parallel dense networks for age and gender prediction
- Output layers:
  - Binary classification for gender (sigmoid activation)
  - Regression for age prediction (sigmoid activation)

Training parameters:
- Batch size: 64
- Epochs: 100 (with early stopping)
- Loss functions: Binary cross-entropy (gender), MAE (age)
- Optimizer: Adam

## Results
The model achieves:
- Gender prediction accuracy: [Add your accuracy here]
- Age prediction MAE: [Add your MAE here]
![image](https://github.com/user-attachments/assets/2de113d7-c9db-4380-bc5d-d765fa7412ec)
![image](https://github.com/user-attachments/assets/247c2012-2e74-44eb-a5b5-756a40947c57)
![image](https://github.com/user-attachments/assets/4448de90-0b8f-450c-a7bd-054fe908e970)
![image](https://github.com/user-attachments/assets/9e78c5f7-0a78-4780-bfef-a826b11a7b7c)
![image](https://github.com/user-attachments/assets/d978d1df-998d-4331-858e-5259ba73d49c)

## Contributing
Contributions to improve the model or add features are welcome:
1. Fork the repository
2. Create a new branch (`git checkout -b feature-branch`)
3. Commit your changes (`git commit -m 'Add new feature'`)
4. Push to the branch (`git push origin feature-branch`)
5. Open a Pull Request
