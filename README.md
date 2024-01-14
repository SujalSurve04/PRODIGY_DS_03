# PRODIGY_DS_03

# Decision Tree Classifier for Iris Dataset
## Overview
This Python script uses the scikit-learn library to implement a Decision Tree Classifier on the famous Iris dataset. The dataset contains measurements of sepal length, sepal width, petal length, and petal width for three different species of iris flowers.

## Dependencies
Make sure you have the necessary libraries installed before running the script. You can install them using:
```bash
pip install scikit-learn
```
### Usage
Clone the repository:
```bash
git clone https://github.com/SujalSurve04/PRODIGY_DS_03
```
### Navigate to the project directory:
```bash
cd your-repository
```
### Run the script:
```bash
python decision_tree_classifier.py
```
Code Explanation
Import Necessary Libraries:

sklearn.datasets: Used to load the Iris dataset.
sklearn.model_selection: Used to split the dataset into training and testing sets.
sklearn.tree.DecisionTreeClassifier: Implements the Decision Tree Classifier.
sklearn.metrics: Includes functions for evaluating the model.
sklearn.tree.export_text: Generates a textual representation of the decision tree.
Load the Iris Dataset:

The Iris dataset is loaded, and features (X) and target labels (y) are extracted.
Split the Dataset:

The dataset is split into training and testing sets using the train_test_split function.
Initialize and Train the Decision Tree Classifier:

A Decision Tree Classifier is initialized and trained on the training set.
Make Predictions:

The trained model is used to make predictions on the test set.
Evaluate the Model:

Accuracy, confusion matrix, and classification report are calculated to evaluate the performance of the model.
Display Results:

The script prints the accuracy, confusion matrix, and classification report.
Display Decision Tree Rules:
The decision tree rules are displayed in a textual format using the export_text function.

### Results
The script outputs the accuracy of the model, confusion matrix, and classification report. Additionally, it provides the decision tree rules in a human-readable format.

### Author
Sujal Surve


