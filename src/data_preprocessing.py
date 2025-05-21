import pandas as pd
from sklearn.preprocessing import StandardScaler

def load_data(omics_paths):
    data_dict = {}
    for omic, path in omics_paths.items():
        df = pd.read_csv(path, index_col=0)
        data_dict[omic] = df
    return data_dict

def normalize_data(data_dict):
    scaler = StandardScaler()
    norm_dict = {}
    for omic, df in data_dict.items():
        df_scaled = pd.DataFrame(
            scaler.fit_transform(df.T).T,
            index=df.index,
            columns=df.columns
        )
        norm_dict[omic] = df_scaled
    return norm_dict

def align_data(data_dict, label_path):
    labels = pd.read_csv(label_path, index_col=0)
    common_samples = set.intersection(*(set(df.columns) for df in data_dict.values()))
    common_samples = list(common_samples & set(labels.index))

    for omic in data_dict:
        data_dict[omic] = data_dict[omic][common_samples]

    y = labels.loc[common_samples, 'Subtype']
    return data_dict, y
