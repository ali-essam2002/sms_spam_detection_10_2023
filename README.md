# sms_spam_detection

# Text Classification with Machine Learning

This project demonstrates text classification using a variety of machine learning algorithms. The primary goal is to classify text messages as either spam or not spam (ham).

## Table of Contents

1. [Introduction](#introduction)
2. [Preprocessing](#preprocessing)
3. [Feature Extraction](#feature-extraction)
4. [Model Selection](#model-selection)
5. [Model Evaluation](#model-evaluation)
6. [Saving the Model](#saving-the-model)

## Introduction

In this project, we build a text classification model to classify text messages into two categories: spam and not spam (ham). The project involves several key steps, including data preprocessing, feature extraction, model selection, and model evaluation.

## Preprocessing

The data preprocessing step is crucial to prepare the text data for machine learning. The following preprocessing steps are performed on the text data:

- Conversion to lowercase.
- Tokenization using NLTK.
- Removal of stopwords.
- Stemming using the Porter Stemmer.

## Feature Extraction

For feature extraction, we utilize the CountVectorizer from scikit-learn to convert the preprocessed text data into a numerical format that machine learning models can understand. This step involves creating a bag of words representation of the text data.

## Model Selection

We experiment with several machine learning algorithms for text classification. The following algorithms are used in this project:

- Support Vector Classifier (SVC)
- K-Nearest Neighbors (KNN)
- Multinomial Naive Bayes (MNB)
- Decision Tree Classifier (DTC)
- Logistic Regression (LR)
- Random Forest Classifier (RFC)
- AdaBoost Classifier (ABC)
- Bagging Classifier (BC)
- Extra Trees Classifier (ETC)
- Gradient Boosting Classifier (GBDT)
- XGBoost Classifier (XGBC)

## Model Evaluation

The performance of each model is evaluated using two metrics: accuracy and precision. These metrics provide insights into the model's ability to classify text messages correctly and its precision in identifying spam.

## Saving the Model

The trained logistic regression model (LR) is saved using the `pickle` library, along with the CountVectorizer (vectorizer) used for feature extraction. These saved models can be used for future text classification tasks.

To run this project, ensure you have the required libraries installed and the dataset (e.g., "spam.csv") in the same directory. You can use the provided code to train and evaluate different models.

Feel free to reach out if you have any questions or need further assistance with this project.

