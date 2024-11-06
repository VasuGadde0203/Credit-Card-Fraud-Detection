import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def load_data(file_path):
    # Load the dataset
    return pd.read_csv(file_path)

def preprocess_data(data):
    # Separate the data for analysis

    legit = data[data['Class'] == 0]
    fraud = data[data['Class'] == 1]
    
    # Sampling to balance the dataset
    legit_sample = legit.sample(n=len(fraud))
    new_data = pd.concat([legit_sample, fraud], axis=0)
    X = new_data.drop(columns='Class', axis=1)
    Y = new_data['Class']
    return X, Y

def scale_features(X):
    # Scaling the 'Amount' column
    scaler = StandardScaler()
    X['Amount'] = scaler.fit_transform(X[['Amount']])
    return X

def apply_pca(X, n_components=10):
    # Applying PCA for dimensionality reduction
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X)
    return X_pca
