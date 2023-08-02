import enum
import typing

import pandas as pd
import numpy as np
import tensorflow as tf
import pickle

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, Normalizer

from scipy import stats
from typing import List


def split_labels_annotated(df):
    """
    Split the dataframe into features and labels.
    Args:
        df: the dataframe to split
        useRaw: whether to use the rawState column or not
    
    Returns:
        x: the features
        y: the labels
    """

    skeep, sdrop1, sdrop2 = 'label', 'state', 'rawState'

    df1 = df.copy()

    x = df1.drop([skeep, sdrop1, sdrop2], axis=1)
    y = df1[skeep]

    return (x, y)

def split_data(df, test_size=0.2, seed=13):
    """
    Split the dataframe into train and test dataframes.
    Args:
        df: the dataframe to split
        test_size: the size of the test dataframe
        seed: the seed for the random state
    
    Returns:
        df_train: the training dataframe
        df_test: the test dataframe
    """
    # Split the data into train and test by mouse, 20% of the mice are in the test set

    mice = df['mouse'].unique()
    np.random.seed(seed)
    np.random.shuffle(mice)
    mice_test = mice[:int(len(mice)*test_size)]
    df_test = df[df['mouse'].isin(mice_test)]
    df_train = df[~df['mouse'].isin(mice_test)]

    # df_train, df_test = train_test_split(df, test_size=test_size, random_state=seed)
    return (df_train, df_test)

def split_data_breeds(df, seed = 13):
    """
    The test dataframe is composed of one mouse of each breed.
    The train dataframe is composed of the remaining mice.
    Args:
        df: the dataframe to split
        seed: the seed for the random state

    Returns:
        df_train: the training dataframe
        df_test: the test dataframe
    """
    # for each breed we take one mouse for the test set
    mice = df['mouse'].unique()
    np.random.seed(seed)
    np.random.shuffle(mice)
    mice_test = []
    for breed in df['breed'].unique():
        # for each breed add a random mouse of the breed to the test set
        mice_test.append(np.random.choice(df[df['breed'] == breed]['mouse'].unique()))
    df_test = df[df['mouse'].isin(mice_test)]
    df_train = df[~df['mouse'].isin(mice_test)]

    return (df_train, df_test)


def prepareLabels(df):
    """
    Create a new column named 'labelstate' in the dataframe.
    Use the rawState to compute the labelstate:
    - 4 and 9 are the wake states
    - 5 nrem
    - 6 rem
    - remove 1,2,3 and s
    """
    df['labelstate'] = df['rawState']
    df['labelstate'] = df['labelstate'].replace([1,2,3,'s'], np.nan)
    df['labelstate'] = df['labelstate'].replace([4,9], 'w')
    df['labelstate'] = df['labelstate'].replace([5], 'n')
    df['labelstate'] = df['labelstate'].replace([6], 'r')
    df = df.dropna()
    return df

# ---------- EPFL project methods ----------
# source : https://github.com/epfl-ML/project2-code
def encode_scale_data(df_train, df_test, seed, cat_matrix):
    """
    Encode the labels, and scale the features.
    Good for using different mice or different days for train and test

    Args:
        df_train: the training dataframe
        df_test: the test dataframe
        useRaw: whether to use the rawState column or not
        seed: the seed for the random state
        cat_matrix: whether to return a categorical matrix or not

    Returns:
        x_train: the training features
        x_test: the test features
        y_train: the training labels
        y_test: the test labels
        le: the label encoder
    """
    x_train, y_train_raw = split_labels(df_train)
    y_train, le = encode_labels(y_train_raw, cat_matrix=cat_matrix)
    

    x_test, y_test_raw = split_labels(df_test)
    y_test = le.transform(y_test_raw)
    if cat_matrix:
        y_test = tf.keras.utils.to_categorical(y_test)

    scaler = StandardScaler().fit(x_train)
    x_train = scaler.transform(x_train)
    x_test = scaler.transform(x_test)

    with open('/home/magali.egger/shared-projects/mice_UNIL/Data/scaler_class_all.pkl', 'wb') as f:
        pickle.dump(scaler, f)

    return x_train, x_test, y_train, y_test, le



def split_labels(df):
    """
    Split the dataframe into features and labels.
    Args:
        df: the dataframe to split
        useRaw: whether to use the rawState column or not
    
    Returns:
        x: the features
        y: the labels
    """

    skeep, sdrop = 'rawState', 'state'

    df1 = df.copy()

    x = df1.drop([skeep, sdrop], axis=1)
    y = df1[skeep]

    return (x, y)


def encode_labels(y, cat_matrix=False):
    """
    Encode the labels.
    Args:
        y: the labels to encode
        cat_matrix: whether to return a categorical matrix or not
    
    Returns:
        y1: the encoded labels
        le: the label encoder
    """

    le = LabelEncoder()
    le.fit(y)
    y1 = le.transform(y)
    if cat_matrix:
        y1 = tf.keras.utils.to_categorical(y1)
    return (y1, le)

def decode_labels(le, y):
    """
    Decode the labels.
    Args:
        le: the label encoder
        y: the labels to decode
    
    Returns:
        y1: the decoded labels
    """

    return le.inverse_transform(y)

    





