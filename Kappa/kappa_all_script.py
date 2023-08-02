

# imports
import pandas as pd
import numpy as np
import sklearn.metrics as metrics
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler

import sys
sys.path.append('../Library')
import breedManip as breedManip
import dataProcessing as dataProcessing
import breeds as breeds
import splitData as splitData

import pickle
import os

# use pickle to load the model
with open('/home/magali.egger/workspace/TBproject/Travail_Bachelor/Pickle/rfc.pkl', 'rb') as f:
    rfc = pickle.load(f)

# get the data for day 0 and 1 of all mice
input_folder = '/home/magali.egger/shared-projects/mice_UNIL/BXD envoie/Prep_files/'
files = [file for file in os.listdir(input_folder)]
df = pd.DataFrame()
for file in files :
    file_name = file.split('.')[0]
    df = pd.concat([df,pd.read_csv(input_folder + file).assign(mouse=file_name)])
df = df.drop(columns=['rawState'])
df = df[(df['day'] == 0) | (df['day'] == 1)]

# split the features and the labels of the df
df = df.drop(columns=['mouse'])
features = df.drop(columns=['state'])
labels = df['state']

# normalize the features
scaler = StandardScaler().fit(features)
features = scaler.transform(features)

# compute the prediction
prediction = rfc.predict(features)
ground_truth, le = splitData.encode_labels(labels)

pred = np.argmax(prediction, axis=1)
 
# prepare the results
cm = confusion_matrix(ground_truth, pred)
kappa = metrics.cohen_kappa_score(ground_truth, pred)

result_path = '/home/magali.egger/workspace/TBproject/Travail_Bachelor/Kappa/result.csv'
with open(result_path, 'a') as f:
    f.write(str(cm) + '  ,  ' + str(kappa) + '\n\n')