import pandas as pd
import numpy as np

import matplotlib
import seaborn
import matplotlib.dates as md
from matplotlib import pyplot as plt

from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.covariance import EllipticEnvelope
from pyemma import msm
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM

def getDistanceByPoint(data, model):
    distance = pd.Series()
    for i in range(0,len(data)):
        Xa = np.array(data.loc[i])
        Xb = model.cluster_centers_[model.labels_[i]-1]
        distance.set_value(i, np.linalg.norm(Xa-Xb))
    return distance

# train markov model to get transition matrix
def getTransitionMatrix (df):
	df = np.array(df)
	model = msm.estimate_markov_model(df, 1)
	return model.transition_matrix

def markovAnomaly(df, windows_size, threshold):
    transition_matrix = getTransitionMatrix(df)
    real_threshold = threshold**windows_size
    df_anomaly = []
    for j in range(0, len(df)):
        if (j < windows_size):
            df_anomaly.append(0)
        else:
            sequence = df[j-windows_size:j]
            sequence = sequence.reset_index(drop=True)
            df_anomaly.append(anomalyElement(sequence, real_threshold, transition_matrix))
    return df_anomaly

def get_data():
    data = pd.read_csv("1be94a28-3bb4-4fd1-86b9-8919a90b7d12.csv")
    temp_data = pd.read_csv("7c638235-b6ad-4b6c-a421-83c094034b43.csv")
    frames = [data,temp_data]
    results = pd.concat(frames,ignore_index=True)
    return results

def add_dummies(data,types):
    for key in types:
        data[key] = 0
    return data

def assign_dummies(data,types_dict):
    for idx,row in data.iterrows():
        temp = data.type.iloc[idx]
        data.iloc[idx,types_dict[temp]] = 1
    return data

def remove_columns(data,columns):
    for key in columns:
        try:
            del data[key]
        except:
            pass
    return data

def scale_data(data):
    X = data.loc[:,:].values
    X = np.array(X)
    for column in range(0,2):
        mu = (sum(X[:,column])*1.0)/X.shape[0]
        maxi = np.max(X[:,column])
        mini = np.min(X[:,column])
        print(maxi,mini)
        for idx in range(0,X.shape[0]):
            try:
                X[idx,column] = (X[idx,column]-mini)/(maxi-mini)
            except:
                pass
    data = pd.DataFrame(data=X,columns=data.columns.values.tolist())
    return X,data
