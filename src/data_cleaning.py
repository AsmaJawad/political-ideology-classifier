import pandas as pd
import numpy as np

#load csv file
def load_data(file : str):
    file_path = f"raw_data/{file}"
    df = pd.read_csv(file_path)
    return df

#clean data
def clean_data(df):
    
    nan_data = [99, 900000] #as labeled in the codebook excel file
    df = df.replace(nan_data, np.nan) #replace these codes with nan

    #keep only needed attributes
    features_map = {
        'IDEO': 'political_ideology',
        'WEIGHT': 'feature_weight',
        'BIRTHHALFDECADE': 'birth_cohort',
        'RACECMB': 'ethnicity',
        'GENDER': 'gender',
        'EDUCREC': 'education_level',
        'FAMILY': 'family_religion',
        'RELPER': 'religiosity',
        'ATTNDPERRLS': 'service_in_person_attendance',
        'ATTNDONRLS': 'service_online_attendance',
        'SCIMPACT': 'science_impact',
        'RELIMPACT': 'religion_impact',
        'SCRLCON1': 'science_vs_religion',
        'FRMREL2': 'childhood_religion',
        'LIFEDIR': 'religiosity_change',
        'SPREL2': 'spouse_religion'
    }

    df = df[list(features_map.keys())].copy()
    df = df.rename(columns=features_map)

    #convert target values into a unified numerical scale
    ideology_mapping = {
        1: 0,  # Very Conservative
        2: 1,  # Conservative
        3: 2,  # Moderate
        4: 3,  # Liberal
        5: 4   # Very Liberal
    }

    df['target_y'] = df['political_ideology'].map(ideology_mapping) #convert ideology ctegories into numerical values
    df.dropna(subset=['target_y'], inplace=True) #drop nan values in target class
    df.drop(columns=['political_ideology'], inplace=True) 

    return df

# if __name__ == "__main__":
#     MY_FILE = "2023-24 RLS Public Use File Feb 19.csv" 
    
#     # 2. load the data
#     raw_df = load_data(MY_FILE)
    
#     # 3. clean it
#     if not raw_df.empty:
#         cleaned_df = clean_data(raw_df)
#         print(f"Original Rows: {len(raw_df)}")
#         print(f"Cleaned Rows:  {len(cleaned_df)}")
#         print(f"Columns:       {len(cleaned_df.columns)}")