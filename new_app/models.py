from django.db import models
from django.db import models

# Create your models here.
from django.db import models

# Create your models here.
from django.db import models
#from keras.models import load_model
#import cv2
import numpy as np
import pickle
#from tensorflow.keras.preprocessing import image
#import tensorflow as tf
import json
from PIL import Image
import pandas as pd 

# Testing phase
rf = pickle.load(open(r"C:\Users\yerra\Music\Loan eligibility prediction\CODING\front end\rf_new.pkl", 'rb'))
xg = pickle.load(open(r"C:\Users\yerra\Music\Loan eligibility prediction\CODING\front end\X_gb_loan.pkl", 'rb'))
knn = pickle.load(open(r"C:\Users\yerra\Music\Loan eligibility prediction\CODING\front end\knn_new.pkl", 'rb'))
voting = pickle.load(open(r"C:\Users\yerra\Music\Loan eligibility prediction\CODING\front end\voting_loan.pkl", 'rb'))


df = pd.read_csv(r'C:\Users\yerra\Music\Loan eligibility prediction\CODING\front end\test.csv')



def predict(algo, row):
    print(row)
    #print(x.columns)  # Assuming `x` is defined
    filter_data = df.iloc[row].values.reshape(1, -1)  # Assuming `df` is defined
    print(filter_data.shape)
    #print(filter_data.columns)  # Assuming `filter_data` has columns

    if algo == 'rf':
        y_pred = rf.predict(filter_data)  # Assuming `rf` is defined
        return y_pred[0]
    elif algo == 'knn':
        y_pred = knn.predict(filter_data)  # Assuming `knn` is defined
        return y_pred[0]
    elif algo == 'voting':
        y_pred = voting.predict(filter_data)  # Assuming `voting` is defined
        return y_pred[0]
    else:
        y_pred = xg.predict(filter_data)  # Assuming `xg` is defined
        return y_pred[0]