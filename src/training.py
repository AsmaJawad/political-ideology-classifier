import pandas as pd
from xgboost import XGBRFClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from data_cleaning import load_data, clean_data

MY_FILE = "2023-24 RLS Public Use File Feb 19.csv"

#sort and plot importance scores for selected features
def feature_importance_plt(rf_model, x):
    
    importance_scrs = rf_model.feature_importances_

    feature_df = pd.DataFrame({
        'Feature': x.columns,
        'Importance': importance_scrs,
    }).sort_values(by='Importance', ascending=True)

    plt.figure(figsize=(10, 6))
    plt.barh(feature_df['Feature'], feature_df['Importance'], color='skyblue')
    plt.xlabel('Importance Score')
    plt.title('Feature Importance: What Drives Political Ideology?')
    plt.tight_layout()
    plt.show()


#random forest model implementation
def train_model():
    
    #load and clean dataset
    raw_data = load_data(MY_FILE)
    df = clean_data(raw_data)

    #define target class, features, and weights from the dataset
    x = df.drop(columns=['target_y', 'feature_weight'])
    y = df['target_y']
    weights = df['feature_weight']

    #encode columns to Panda's categorial type -> helps xgboost handle Nan vals
    for col in x.columns:
        x[col] = x[col].astype('category').cat.codes

    #split the dataset between training and testing
    x_train, x_test, y_train, y_test, w_train, w_test = train_test_split(
        x, y, weights, test_size=0.2, random_state=42
        )

    #configure xgboost for random forest ML model implementation
    rf_model = XGBRFClassifier(
        n_estimators=1,
        num_parallel_tree=100,
        subsample=0.8,
        colsample_bynode=0.8,
        random_state=42,
        use_label_encoder=False,
        eval_metric='mlogloss'
    )

    #model fitting
    rf_model.fit(x_train, y_train, sample_weight=w_train)
    y_pred = rf_model.predict(x_test)
    
    #plot confusion matrix for model performance analysis
    def conf_matrix_plot(y_test, y_pred):
        raw_cts = confusion_matrix(y_test, y_pred)

        labels = ['very_conservative', 'conservative', 'moderate', 'liberal', 'very_liberal']

        disp = ConfusionMatrixDisplay(confusion_matrix=raw_cts, display_labels=labels)
        disp.plot(cmap=plt.cm.Blues)
        plt.title("Confusion Matrix: Political Ideology")
        plt.show()
        
    print("\n--- Model Evaluation ---")
    print(classification_report(y_test, y_pred))
    conf_matrix_plot(y_test, y_pred)
    feature_importance_plt(rf_model, x_train)

if __name__ == "__main__":
    train_model()

