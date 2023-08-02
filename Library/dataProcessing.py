import pandas as pd
import numpy as np
import enum
import typing
from typing import List



def addDayNumber(df) -> pd.DataFrame:
    """
    Add the day number to the dataframe
    Args:
        df: the dataframe to add the day number to
    Returns:
        dataframe: the dataframe with the day number added
    """
    dataframe = df.copy()
    dataframe["epoch"] = dataframe.index
    dataframe["day"] = dataframe.index // 21600
    # dataframe["hour"] = dataframe.index // 900
    # dataframe["minute"] = dataframe.index // 15
    return dataframe


def addAllFeatures(dataFolder, dataFiles, dropBins = True, ) -> pd.DataFrame:
    """
    1. for all files
        1. pd.read_csv() all files in dataFiles
        2. add DayNumber
        3. add flatness
        4. add centroid
        5. add entropy
        6. drop the bins 
        7. add rolling windows
        8. remove outliers
        9. non linearity
        10. concat
    2. rebalance
    3. remove temp
    """
    
    windows_sizes = [2, 5, 10, 20, 50, 100]
    dfComplete = pd.DataFrame()
    seed = 13
    rolloffs = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    
    for file in dataFiles :
        df = pd.read_csv(dataFolder + file +'.smo.csv')
        df = addDayNumber(df)

        df = spectral_flatness(df)
        df = spectral_centroid(df)
        df = spectral_entropy(df)

        for rolloff in rolloffs:
            df = spectral_rolloff(df, p=rolloff)

        for i in range(401):
            df = df.drop([f"bin{i}"], axis=1)
        
        df = add_feature_windows(df, windows_sizes, ["EEGv", "EMGv"])

        df = remove_outliers_quantile(df, ["EEGv", "EMGv"])

        df = non_linearity(df, ["EEGv", "EMGv"])

        dfComplete = pd.concat([dfComplete, df])
    
    # dfComplete = rebalance_state(dfComplete, seed)
    dfComplete = dfComplete.drop(["temp"], axis=1)

    return dfComplete


# Methods from the epfl project ---------------------------------------------------------------
# source : https://github.com/epfl-ML/project2-code

def spectral_flatness(dataframe):
    """
    Calculate the spectral flatness of the dataframe
    Args:
        dataframe: the dataframe to calculate the spectral flatness of
    Returns:
        dataframe: the dataframe with the spectral flatness column added
    """

    df = dataframe.copy()
    bins = [f"bin{i}" for i in range(401)]

    # sum log of bins
    sum_log = df[bins].apply(lambda x: np.log(x), axis=1).sum(axis=1)
    # divide by number of bins
    mean_log = sum_log / 401
    # exponentiate
    exp = mean_log.apply(lambda x: np.exp(x))
    # divide by sum
    res =  401 * (exp / df[bins].sum(axis=1))

    df['spectral_flatness'] = res
    return df


def spectral_centroid(dataframe):
    """
    Calculate the spectral centroid of the dataframe
    Args:
        dataframe: the dataframe to calculate the spectral centroid of
    Returns:
        dataframe: the dataframe with the spectral centroid column added
    """

    df = dataframe.copy()
    bins = [f"bin{i}" for i in range(401)]

    # weighted sum
    weighted_sum = df[bins].apply(lambda x: np.sum(x * np.arange(401) * 0.25), axis=1)
    sum = df[bins].sum(axis=1)
    
    df['spectral_centroid'] = weighted_sum / sum

    return df

def spectral_entropy(dataframe):
    """
    Calculate the spectral entropy of the dataframe
    Args:
        dataframe: the dataframe to calculate the spectral entropy of
    Returns:
        dataframe: the dataframe with the spectral entropy column added
    """
    
    df = dataframe.copy()
    bins = [f"bin{i}" for i in range(401)]

    # normalize bins
    df2 = df[bins].apply(lambda x: x / x.sum(), axis=1)
    
    def entropy(x):
        return np.sum(x * np.log2(x))

    # calculate entropy
    df['spectral_entropy'] = df2.apply(lambda x: entropy(x), axis=1)

    return df


def spectral_rolloff(dataframe, p):
    """
    Calculate the spectral rolloff of the dataframe
    Args:
        dataframe: the dataframe to calculate the spectral rolloff of
        p: the percentage of the spectral rolloff
    Returns:
        dataframe: the dataframe with the spectral rolloff column added
    """

    df = dataframe.copy()

    bins = [f"bin{i}" for i in range(401)]

    df[f'spectral_rolloff_{p}'] = df[bins].apply(lambda x: np.argmax(x.cumsum() >= x.sum() * p) * 0.25, axis=1)
    
    return df


class WindowOperationFlag(enum.IntFlag):
    """
    The different features that can be extracted from a window. If multiple features are selected,
    they will all be extracted from the window.
    """

    MEAN = enum.auto()
    MEDIAN = enum.auto()
    VAR = enum.auto()
    MIN = enum.auto()
    MAX = enum.auto()


def features_window(
        df: pd.DataFrame,
        window_size: int,
        op: WindowOperationFlag = 0,
        features: typing.Union[List[str], None] = None,
        center=False,
) -> pd.DataFrame:
    """
    Smooth features by performing a set of operations over a window of size `window_size`. The implementations must
    select the features to smooth from the `features` argument. If `features` is None, all features will be smoothed.
    Args:
        df: numpy array of shape (N, D).
        window_size: size of the window.
        op: the operation(s) to perform on the window.
        features: the list of features
        center: whether the window is centered or not.

    Returns:
        data: numpy array of shape (N, D).
    """

    if op == 0:
        op = WindowOperationFlag.MEAN | WindowOperationFlag.MEDIAN | WindowOperationFlag.VAR | WindowOperationFlag.MIN | WindowOperationFlag.MAX
    if features is None:
        features = list(df.columns)

    # For each operation, add the computed aggregated value.
    if WindowOperationFlag.MEAN & op == WindowOperationFlag.MEAN:
        df[[f + f"_mean{window_size}" for f in features]] = df[features].rolling(window_size, center=center).mean()
    if WindowOperationFlag.MEDIAN & op == WindowOperationFlag.MEDIAN:
        df[[f + f"_median{window_size}" for f in features]] = df[features].rolling(window_size, center=center).median()
    if WindowOperationFlag.VAR & op == WindowOperationFlag.VAR:
        df[[f + f"_var{window_size}" for f in features]] = df[features].rolling(window_size, center=center).var()
    if WindowOperationFlag.MIN & op == WindowOperationFlag.MIN:
        df[[f + f"_min{window_size}" for f in features]] = df[features].rolling(window_size, center=center).min()
    if WindowOperationFlag.MAX & op == WindowOperationFlag.MAX:
        df[[f + f"_max{window_size}" for f in features]] = df[features].rolling(window_size, center=center).max()

    return df

def add_feature_windows(df, window_sizes, window_features):
    """
    Add rolling windows for each window size.
    Args:
        df: the dataframe to transform
        window_sizes: the list of window sizes
        window_features: the list of features to apply the window to
    
    Returns:
        df: the transformed dataframe
    """

    window_names = ['EEGv', 'EMGv']
    for window_size in window_sizes:
        df = features_window(df, window_size=window_size, op=WindowOperationFlag.MEAN, features=window_features)
        df = features_window(df, window_size=window_size, op=WindowOperationFlag.MEDIAN, features=window_features)
        df = features_window(df, window_size=window_size, op=WindowOperationFlag.VAR, features=window_features)
        df = features_window(df, window_size=window_size, op=WindowOperationFlag.MIN, features=window_features)
        df = features_window(df, window_size=window_size, op=WindowOperationFlag.MAX, features=window_features)
        for feature in window_features:
            window_names.append(feature + "_mean" + str(window_size))
            window_names.append(feature + "_median" + str(window_size))
            window_names.append(feature + "_var" + str(window_size))
            window_names.append(feature + "_min" + str(window_size))
            window_names.append(feature + "_max" + str(window_size))

    # drop nan from feature window
    df = df.dropna()

    return df


def remove_outliers_quantile(dataframe, my_features, threshold=0.99):
    """
    Remove outliers from dataframe
    Args:
        df: dataframe with features
        features: list of features to remove outliers from
        threshold: threshold for quantile
    Returns:
        df: dataframe without outliers
    """
    df = dataframe.copy()

    alpha = 1 - threshold
    for feature in my_features:
        q_lower = df[feature].quantile(alpha / 2)
        q_upper = df[feature].quantile(1 - alpha / 2)
        df = df[(df[feature] > q_lower) & (df[feature] < q_upper)]

    return df


def non_linearity(df, features=[]):
    """
    Calculate the non-linearity of the features.
    Args:
        df: the dataframe to transform
        features: the list of features to calculate the non-linearity of

    Returns:
        df: the transformed dataframe
    """
    df = log_features(df, features=features)
    df = expand_features_poly(df, max_degree=3, features=features)

    return df


def log_features(df, features=[]):
    """
    Take the log of the features.
    Args:
        df: the dataframe to transform

    Returns:
        df: the transformed dataframe
    """

    df1 = df.copy()

    # drop zeroes
    size_before = df1.shape[0]
    for feature in features:
        df1 = df1[df1[feature] > 0]
    size_after = df1.shape[0]
    if size_before != size_after:
        print(f"Removed {size_before - size_after} rows with invalid log values in {file}")

    # apply log
    for feature in features:
        df1[f"{feature}_log"] = np.log(df1[feature])
    return df1


def expand_features_poly(dataframe, max_degree, features=None):
    """
    Expand the dataframe by adding polynomial features.

    Args:
        dataframe: the dataframe to expand
        max_degree: maximum degree of the polynomial
        features: the list of features to expand

    Returns:
        df: the expanded dataframe
    """

    df = dataframe.copy()

    # add bias
    df["bias"] = 1
    
    if features is None:
        features = ['EEGv', 'EMGv']

    for feature in features:
        for degree in range(2, max_degree + 1):
            df[f"{feature}^{degree}"] = df[feature] ** degree

    return df

def rebalance_state(df, seed, label_column="state"):
    """
        Rebalance the labels in the dataframe
        Args:
            df: dataframe with labels
            label_column: column with labels
        Returns:
            df: dataframe with balanced labels
    """

    balance = df[label_column].value_counts().min()
    df = df.groupby(label_column).apply(lambda x: x.sample(balance, random_state=seed)).reset_index(drop=True)
    return df


def encode_state(state):
    """
        Encode the state to a number
        Args:
            state: the state to encode
        Returns:
            encoded state
    """
    if state == "w":
        return 0
    elif state == "n":
        return 1
    elif state == "r":
        return 2
    else:
        return -1