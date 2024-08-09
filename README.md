# Machine Learning - Stroke Classification


## Tabel of Content

 - [Introduction](#Introduction)
 - [Project Structure](#Project-Structure)
 - [Installation](#Installation)
 - [Dataset](#Dataset)
 - [Feature Correlation](#Feature-Correlation)
 - [Model Training](#Model-Training)
 - [Results](#Results)
 - [Contact](#Contact)

## Introduction

This project aims to classify whether a patient is likely to have a stroke based on various health parameters using machine learning techniques. The classification model is built using Logistic Regression, Random Forest Classifier, etc. The dataset used contains information such as age, gender, hypertension, heart disease, and other health-related features.

## Project Structure
```bash
stroke-classification/
├── data/                   		    # Data files
├── images/                   		    # Saved evaluate
├── models/                   		    # Saved models
├── src/                      		    # Source code for the project
│   ├── config.py           		    # Script for configuration settings
│   ├── stroke_classification.py            # Script for training models
├── requirements.txt          		    # Python packages required
├── README.md                 		    # Project documentation
```

## Installation

To set up the project, clone this repository and install the necessary dependencies:
```bash
 git clone https://github.com/ndanh318/Stroke-Classification.git
 cd Stroke-Classification
 pip install -r requirements.txt
```
## Dataset

Here is a preview of the stroke classification dataset. The dataset used for this project can be found [here](https://github.com/ndanh318/Stroke-Classification/tree/master/data). 
| pat_id | stroke | gender | age | hypertension | heart_disease | work_related_stress | urban_residence | avg_glucose_level | bmi | smokes |
 |--------|--------|--------|------|--------------|---------------|--------------------|-----------------|-------------------|-------|--------| 
 | 1 | 1 | Male | 67.0 | 0 | 1 | 0 | 1 | 228.69 | 36.6 | 1 |
 | 2 | 1 | Female | 61.0 | 0 | 0 | 1 | 0 | 202.21 | NaN | 0 | 
 | 3 | 1 | Male | 80.0 | 0 | 1 | 0 | 0 | 105.92 | 32.5 | 0 | 
 | 4 | 1 | Female | 49.0 | 0 | 0 | 0 | 1 | 171.23 | 34.4 | 1 | 
 | 5 | 1 | Female | 79.0 | 1 | 0 | 1 | 0 | 174.12 | 24.0 | 0 |

-   **Features**:
    
    -	`pat_id`: Patient ID
    -   `gender`: Gender of the patient.
    -   `age`: Age of the patient.
    -   `hypertension`: Whether the patient has hypertension (0: No, 1: Yes).
    -   `heart_disease`: Whether the patient has heart disease (0: No, 1: Yes).
    -   `work_related_stress`: Indicator of work-related stress (0: No, 1: Yes).
    -   `urban_residence`: Whether the patient lives in an urban area (1: Yes, 0: No).
    -   `avg_glucose_level`: Average glucose level in the blood.
    -   `bmi`: Body Mass Index.
    -   `smokes`: Smoking status of the patient (1: Yes, 0: No).

-   **Target**:
    
    -   `stroke`: Indicates whether the patient has had a stroke (1: Yes, 0: No).

## Feature Correlation

The heatmap below shows the correlations between different features in the stroke classification dataset. The values range from -1 (strong negative correlation) to 1 (strong positive correlation).

![Correlation Heatmap](https://github.com/ndanh318/Stroke-Classification/blob/master/images/correlation.png)

## Model Training

To train the model, use the following command:
```bash
python src/stroke_classification.py
```

## Results
### Logistic Regression
![Logistic Regression](https://github.com/ndanh318/Stroke-Classification/blob/master/images/logistic_regression.png)

The model achieved the following metrics on the test set:
- Accuracy score: 0.54
- Precision score: 0.11
- Recall score: 0.95
### Random Forest
![Random Forest](https://github.com/ndanh318/Stroke-Classification/blob/master/images/random_forest.png)

The model achieved the following metrics on the test set:
- Accuracy score: 0.87
- Precision score: 0.22
- Recall score: 0.42
- ### Decision Tree
![Decision Tree](https://github.com/ndanh318/Stroke-Classification/blob/master/images/decision_tree.png)

The model achieved the following metrics on the test set:
- Accuracy score: 0.86
- Precision score: 0.17
- Recall score: 0.31
- ### k-Nearest Neighbors
![k-Nearest Neighbors](https://github.com/ndanh318/Stroke-Classification/blob/master/images/k-nearest_neighbors.png)

The model achieved the following metrics on the test set:
- Accuracy score: 0.74
- Precision score: 0.14
- Recall score: 0.65

## Contact

For any questions or issues, please contact me at ngoduyanh8888@gmail.com.

