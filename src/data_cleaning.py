import pandas as pd
import numpy as np

#load csv file
def load_data(file : str):
    file_path = f"raw_data/{file}"
    df = pd.read_csv("{file_path}")
    return df

#clean data
def clean_data(df):
    nan_data = {99, 900000} #as labeled in the codebook excel file
    df = pd.replace(nan_data, np.nan) #replace these codes with nan

    ideology_mapping = {
        1: 0,  # Very Conservative
        2: 1,  # Conservative
        3: 2,  # Moderate
        4: 3,  # Liberal
        5: 4   # Very Liberal
    }

    df['target_y'] = df['IDEO'].map(ideology_mapping) #convert ideology ctegories into numerical values
    df.dropna(subset=['target_y'], inplace=True) #drop nan values

    return df

