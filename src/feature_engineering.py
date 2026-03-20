import pandas as pd

def add_time_features(df):
    df['month'] = df['Date'].dt.month
    df['day_of_week'] = df['Date'].dt.dayofweek
    return df

def add_season(df):
    def get_season(month):
        if month in [12,1,2]:
            return 'Winter'
        elif month in [3,4,5]:
            return 'Summer'
        elif month in [6,7,8,9]:
            return 'Monsoon'
        else:
            return 'Post-Monsoon'

    df['season'] = df['month'].apply(get_season)
    return df

def add_traffic(df):
    df['traffic_level'] = df['day_of_week'].apply(
        lambda x: 'High' if x < 5 else 'Low'
    )
    return df

def add_industry(df):
    df['industrial_activity'] = df['season'].apply(
        lambda x: 1 if x == 'Winter' else 0
    )
    return df

def add_lag_features(df):
    df['AQI_prev_1'] = df['AQI'].shift(1)
    df['AQI_prev_2'] = df['AQI'].shift(2)
    df['AQI_prev_3'] = df['AQI'].shift(3)
    return df

def final_processing(df):
    df = df.dropna()
    df = df.drop(['City'], axis=1)
    df = pd.get_dummies(df, columns=['season','traffic_level'], drop_first=True)
    return df