# signal_processing-audio-classification
Neural Network Model on Classification (on/off ) of Jet engine sounds aimed at improving cost-effective fleet management

# Aim

The primary goal of this project is to develop an audio classification model using TensorFlow and Keras. The model aims to classify audio samples as either True (1) or False (0) based on specific sound patterns detected in the audio data.

## Data

The project utilizes audio data collected from various sources. Each audio sample is labeled as either True (1) or False (0) based on the presence of target sound patterns. The data collection process involves reading audio files and extracting relevant features for training the classification model.

## Files in the Repository

The repository contains the following key files and resources:

- `README.md`: This documentation file providing an overview of the project.
- jet-engine-audio-classification.ipynb: notebook for data preprocessing, model development, and evaluation.
- jet-engine-audio-classification.py: python script for data preprocessing, model development, and evaluation.

## Dependencies

To run the project successfully, you need to install the following Python packages:

- `os`: For operating system-related functions.
- `numpy`: For numerical operations and array handling.
- `pandas`: For data manipulation and handling DataFrames.
- `matplotlib.pyplot`: For data visualization.
- `tensorflow`: For deep learning model development.
- `tensorflow_io`: For audio data handling.
- `keras_tuner`: For hyperparameter tuning.
- `librosa`: For audio feature extraction.
- `IPython.display`: For displaying audio samples.
- `glob`: For file path manipulation.
- `tqdm`: For progress bars and monitoring loops.
- `seaborn`: For enhanced data visualization.
- `itertools.cycle`: For cycling through color palettes.
- `sklearn.model_selection.train_test_split`: For data splitting.
- `sklearn.metrics.accuracy_score`: For accuracy calculation.
- `sklearn.metrics.recall_score`: For recall calculation.
- `imblearn.under_sampling.ClusterCentroids`: For addressing data class imbalance in undersampling.
- `tensorflow.keras.models.Sequential`: For building the neural network model.
- `tensorflow.keras.layers`: For adding layers to the model.
- `datetime`: For measuring training duration.

Please make sure to set up a virtual environment and install these packages to avoid conflicts with system packages.

## Handling Data Imbalance

The project addresses class imbalance in the training data using undersampling techniques. This step ensures that the model does not become biased toward the majority class and can effectively classify both True and False samples.

## Model Design

The audio classification model is designed using TensorFlow and Keras. The architecture includes neural network layers for feature extraction and classification. Hyperparameter tuning is performed to optimize the model's performance.

## Model Results

The model is evaluated on a test dataset, and the following results are obtained:

- Best validation accuracy achieved during training: 0.9286
- Test accuracy: 1.0 (perfect accuracy)

These results demonstrate the effectiveness of the classification model in accurately identifying True and False audio samples.

## Possible Modifications/Improvements

While the current model achieves high accuracy, there is always room for improvement. Potential modifications and enhancements for the project include:

- Exploring more complex neural network architectures.
- Incorporating additional audio features for improved classification.
- Experimenting with different undersampling and data augmentation techniques.
- Fine-tuning hyperparameters for further optimization.
- Scaling the model for deployment on mobile devices or web applications.
