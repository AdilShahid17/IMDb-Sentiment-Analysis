# IMDb Sentiment Analysis
This Jupyter notebook provides a comprehensive guide to performing sentiment analysis on the IMDB movie reviews dataset using Recurrent Neural Networks (RNNs), specifically utilizing Long Short-Term Memory (LSTM) units. The project is structured to walk you through the process of data preprocessing, model building, training, and evaluation, leveraging PyTorch and other essential Python libraries.

## Project Structure
### Import Necessary Libraries and Files
The notebook begins by importing necessary Python libraries such as numpy, pandas, matplotlib, seaborn, torch, torchvision, sklearn, and nltk, among others. These libraries are used for data manipulation, visualization, model building, and evaluation.

### Data Loading
The IMDB dataset is loaded from a Google Drive location into a pandas DataFrame. This dataset consists of 50,000 movie reviews labeled as positive or negative. The notebook demonstrates how to mount a Google Drive and read the dataset directly from it.

### Data Preprocessing
Dataset Splitting: The dataset is split into training, validation, and test sets to ensure the model is trained, validated, and tested on different subsets of data.
Tokenization and Vocabulary Building: The text data (movie reviews) are tokenized using nltk's word_tokenize method. A vocabulary is built with a maximum size limit, and GloVe word embeddings are used to represent the words.
BucketIterator: To efficiently handle variable-length text, the BucketIterator from torchtext is used to create iterators for the train, validation, and test sets, ensuring that each batch of data has reviews of similar lengths to minimize padding.

### Model Architecture
An RNN model is defined with the following components:

Embedding Layer: Maps words to their corresponding embeddings.
LSTM Layer: Processes the sequence of word embeddings and captures the temporal dependencies between words.
Fully Connected Layer: Outputs the final sentiment prediction based on the LSTM layer's output.
Dropout: Applied to prevent overfitting.
Training the RNN
The training process involves iterating over the training set, computing the loss (Binary Cross Entropy with Logits Loss), and updating the model parameters. The Adam optimizer is used for optimization. The validation set is used to evaluate the model after each epoch, and the test set is used to evaluate the model's final performance.

### Hyperparameter Tuning
A simple approach to hyperparameter tuning is demonstrated, where different combinations of embedding dimensions, hidden dimensions, learning rates, batch sizes, and weight decays are experimented with to find the best set of hyperparameters based on validation loss.

### Evaluation
The model's performance is evaluated using accuracy as the metric. The baseline model, which predicts sentiments randomly based on the training set's class distribution, is also evaluated for comparison.

### Dataset Shift Experiment
An experiment is conducted to test the model's robustness to dataset shift. A new version of the dataset is created by removing words starting with specific letters from the reviews, simulating a scenario where the test data distribution differs from the training data. The model trained on the original dataset is then evaluated on this shifted dataset to observe the impact on performance.

## How to Use
Setup: Ensure you have the necessary libraries installed in your Python environment. If you're using Google Colab, most of these libraries will be pre-installed.
Data: Upload the IMDB dataset to your Google Drive and adjust the file path in the notebook to point to your dataset location.
Run: Execute the cells in order, from top to bottom. You may tweak hyperparameters or model architecture as desired.
Experiment: To test the model's performance on dataset shifts, modify the function that alters the dataset and observe the impact on model accuracy.

## Dependencies
Python 3.x
numpy
pandas
matplotlib
seaborn
PyTorch
torchvision
scikit-learn
nltk
torchtext
