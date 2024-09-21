# MNIST Classification using LeNet-5 with TensorFlow

This project implements a LeNet-5 Convolutional Neural Network (CNN) for classifying the MNIST dataset of handwritten digits using TensorFlow. The model is enhanced with regularization techniques like Dropout and L2 Regularization to improve generalization and prevent overfitting.

## Requirements

Before running the code, ensure you have the following dependencies installed:

- Python 3.x
- TensorFlow
- NumPy
- Matplotlib
- scikit-learn
- TensorFlow Datasets

You can install the necessary libraries using the following:

```bash
pip install tensorflow numpy matplotlib scikit-learn tensorflow-datasets
```

## Dataset

The MNIST dataset is automatically downloaded using the TensorFlow Datasets API. It contains 60,000 training images and 10,000 test images, each representing a 28x28 grayscale image of a handwritten digit (0-9).

## Model Architecture

This project uses a classic **LeNet-5** architecture with the following layers:
1. **Conv2D**: 6 filters, 5x5 kernel, Tanh activation, L2 Regularization
2. **Average Pooling**: 2x2 pooling
3. **Conv2D**: 16 filters, 5x5 kernel, Tanh activation, L2 Regularization
4. **Average Pooling**: 2x2 pooling
5. **Flatten**: Converts 2D feature maps to 1D
6. **Dense**: 120 units, Tanh activation, L2 Regularization
7. **Dropout**: 50% to prevent overfitting
8. **Dense**: 84 units, Tanh activation, L2 Regularization
9. **Dropout**: 50% to prevent overfitting
10. **Dense (Output)**: 10 units (for 10 classes), Softmax activation

## Features

- **Normalization**: All pixel values are normalized to [0, 1].
- **Regularization**: L2 Regularization and Dropout layers to reduce overfitting.
- **Early Stopping**: The training process stops when no further improvement is observed in validation loss.

## Model Training

The model is trained for up to 20 epochs with early stopping enabled, which monitors the validation loss and stops training after 3 epochs of no improvement.

## Evaluation

After training, the following metrics and visualizations are provided:

1. **Accuracy and Loss Curves**: Plots of training and validation accuracy and loss over epochs.
2. **Confusion Matrix**: Visual representation of the classification performance across the 10 digit classes.
3. **ROC Curve**: ROC curve for each digit class, showing the trade-off between true positive and false positive rates.
4. **Test Accuracy**: Overall accuracy of the model on the test dataset.

## How to Run

1. Clone the repository:

```bash
git clone https://github.com/ForresterDaniel/MNIST_LeNet-5.git
cd MNIST_LeNet-5
```

2. Run the Python script:

```bash
python MNIST_LeNet-5.ipynb
```

The script will download the dataset, train the model, and display the evaluation metrics.

## Results

The model achieves a test accuracy of around **98.5%** on the MNIST dataset.

## License

GNU General Public License v3 (GPL-3.0)
