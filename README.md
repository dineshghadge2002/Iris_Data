# Iris Flower Species Classification

This repository contains a Python script for classifying Iris flower species using logistic regression.

## Table of Contents

- [Introduction](#introduction)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)
- [Code Overview](#code-overview)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Introduction

This project focuses on classifying Iris flower species based on various features using logistic regression. The Iris dataset contains information about the dimensions of the sepals and petals of three different Iris species.

## Prerequisites

Before you begin, ensure you have the following installed:

- Python (version 3.1)
- Pandas
- Matplotlib
- Seaborn
- NumPy
- Scikit-Learn

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/dineshghadge2002/Iris_Data.git
   cd iris-flower-classification
   ```

2. Install the required dependencies:

   ```bash
   pip install pandas matplotlib seaborn numpy scikit-learn
   ```

## Usage

1. Place the `IRIS (1).csv` dataset file in the project directory.

2. Run the Python script `iris_classification.py` to perform the following tasks:
   - Import necessary libraries.
   - Load the Iris dataset.
   - Perform label encoding for the 'species' column.
   - Select features and target variable.
   - Split the dataset into training and testing sets.
   - Train a logistic regression model.
   - Print model coefficients and intercept.
   - Predict Iris species and calculate accuracy.
   - Print a confusion matrix.

3. Review the script's output, including the model's accuracy and confusion matrix.

## Code Overview

The Python script performs the following steps:
- Imports necessary libraries.
- Loads and inspects the Iris dataset.
- Encodes the 'species' column using label encoding.
- Selects features and target variable.
- Splits the dataset into training and testing sets.
- Trains a logistic regression model.
- Prints the model's coefficients and intercept.
- Predicts Iris species and calculates accuracy.
- Prints a confusion matrix to evaluate the model's performance.

## Results

The logistic regression model achieved an accuracy of 96% on the test set. The confusion matrix provides insights into the model's performance, showing true positives, true negatives, false positives, and false negatives.

## Contributing

Contributions are welcome! If you find any issues or have suggestions for improvement, please feel free to submit a pull request.
