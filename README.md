# Political Ideology Classifier (XGBoost Random Forest)

## Objective 
This project analyzes the 2023-24 Pew Research Center Religious Landscape Study ($N=36,908$) to determine if demographic and religious variables can accurately predict a respondent's political ideology.

## Key Technical Implementations
- Modular Data Pipeline: Engineered a robust data_cleaning.py script to handle high-cardinality survey codes and standardize missing data as np.nan.
- Parallel Ensemble Modeling: Utilized XGBRFClassifier to implement a Random Forest through parallel tree construction, rather than traditional sequential boosting.
- Statistical Integrity: Integrated feature_weights into the model's .fit() procedure to ensure results remain representative of the U.S. population.

## Quick Start & Data Setup
Due to the dataset's large size, I hid it from the public repository. To run this model locally, follow these steps to acquire the dataset and prepare your environment:

### Acquire the Data:
1. Download the dataset zip file provided in <a href="https://www.pewresearch.org/dataset/2023-24-religious-landscape-study-rls-dataset/">Pew Research Dataset Page</a>. Extract the dataset from the csv file "2023-24 RLS Public Use File Feb 19.csv"
2. Create a folder named raw_data/ in the root directory and place the CSV inside it.

### Install Dependencies:
```pip install -r requirements.txt```

### Run the Pipeline:
```python src/training.py``` 

## Current Findings
The initial model achieves an accuracy of 44% across 5 classes. Feature importance analysis reveals that religious attendance is the primary driver of the model's logic, though it currently faces a "Centrist Magnet" challenge where Liberals and Conservatives are often misclassified as Moderates.

## Future Plans
Currently, I am working on implementing the following:
1. Addressing the centrist magnet issue to enhance model accuracy.
3. Containerizing the repository for easier replication of the model.
4. Freezing trained model using the joblib Python library.
5. Creating an interactive website for an in depth analysis of the relationship between religious and political affiliation amongst the US population. For easier implementation of this step, I plan to use Flask framework and Bootstrap for frontend development.
6. Deployment!! :partying_face:

## Graphs

<p align="center">
  <img src="reports/rf_confusion_matrix.png" width="600" title="Feature Importance Plot">
  <br>
  <em>Figure 1: Analysis of model performance and accuracy.</em>
</p>

<p align="center">
  <img src="reports/feature_importance.png" width="600" title="Feature Importance Plot">
  <br>
  <em>Figure 2: Analysis of predictive features for political ideology.</em>
</p>
