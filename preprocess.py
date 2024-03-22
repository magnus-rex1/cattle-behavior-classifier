
import numpy as np
from sklearn.impute import SimpleImputer

# converting from mg to m/s^2
def convert_acc_units(df):
    df['AccX'] = (df['AccX']/1000)*9.81
    df['AccY'] = (df['AccY']/1000)*9.81
    df['AccZ'] = (df['AccZ']/1000)*9.81
    df['AccMag'] = np.sqrt(df['AccX']**2 + df['AccY']**2 + df['AccZ']**2)

    return df

def simple_impute(df):
    mode = SimpleImputer(strategy='most_frequent')
    # df['Label'] = mode.fit_transform(df[['Label']])
    df.loc[:,'Label'] = mode.fit_transform(df[['Label']])

    return df

# encode 'Label column
def encode_label_column(df):
    behaviors = {
    'RES': 1,  # Resting in standing position
    'RUS': 2,  # Ruminating in standing position
    'MOV': 3,  # Moving
    'GRZ': 4,  # Grazing
    'SLT': 5,  # Salt licking
    'FES': 6,  # Feeding in stanchion
    'DRN': 7,  # Drinking
    'LCK': 8,  # Licking
    'REL': 9,  # Resting in lying position
    'URI': 10,  # Urinating
    'ATT': 11, # Attacking
    'ESC': 12, # Escaping
    'BMN': 13, # Being mounted
    'ETC': 14, # Other behaviors
    'BLN': 15  # Data without video, no label
    }
    df['behavior'] = df['Label'].map(behaviors)
    df['behavior'] = df['behavior'].astype('int')
    # preserve the nans in the original column
    # df['behavior'] = df['behavior'].where(~df['Label'].isnull(), pd.NA)

    return df