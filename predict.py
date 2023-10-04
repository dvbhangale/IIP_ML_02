import joblib
import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OrdinalEncoder


def encoder(input_val): 
    with open('Models/encoder.pickle', "rb") as f:
        encoder = pickle.load(f)
    input = encoder.transform(input_val)    
    return input


def get_prediction(data,model):
    target=['above limit', 'below limit']
    res = model.predict(data)
    
    return target[res]
