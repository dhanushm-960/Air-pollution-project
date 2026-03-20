import pandas as pd

def load_data(path):
    return pd.read_csv(path)

def filter_city(df):
    return df[df['City'] == 'Delhi']

def clean_data(df):
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date')
    return df