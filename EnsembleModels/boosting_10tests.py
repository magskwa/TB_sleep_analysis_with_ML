import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.ensemble import EasyEnsembleClassifier
import sys
np.set_printoptions(threshold=sys.maxsize)
import sys
sys.path.append('../Library')
import breedManip as breedManip
import dataProcessing as dataProcessing
import breeds as breeds
import splitData as splitData


# Get the data of all the mice for the 3 days
with open('/home/magali.egger/workspace/TBproject/Travail_Bachelor/Data/df_simplify_day3.pkl', 'rb') as f:
    df = pickle.load(f)
df['breed'] = df['mouse'].apply(lambda x: breedManip.getBreedIndex(breedManip.getBreedOfMouse(x)))

# Keep only the breeds that have at least 4 mice
selected_breeds = breedManip.selectAllBreedsOfSizeNOrMore(4)
id_selected_breeds = [breedManip.getBreedIndex(breed) for breed in selected_breeds]
df = df[df['breed'].isin(id_selected_breeds)]

for i in range(10):
    seed = i

    # Split the data into train and test set
    df_train, df_test = splitData.split_data_breeds(df, seed)
    mice_test = df_test['mouse'].unique()
    df_train = df_train.drop(columns=['mouse'])
    df_test = df_test.drop(columns=['mouse'])
    df_train = df_train.drop(columns=['breed'])
    df_test = df_test.drop(columns=['breed'])
    x_train, x_test, y_train, y_test, le = splitData.encode_scale_data(df_train, df_test, seed, cat_matrix=True)

    # Random forest classifier
    eec = EasyEnsembleClassifier(random_state=13, n_jobs=-1)
    y_train_raw = np.argmax(y_train, axis=1)
    eec.fit(x_train, y_train_raw)
    pred = eec.predict(x_test)

    # Confusion matrix
    test = np.argmax(y_test, axis=1)
    cm = np.array(confusion_matrix(test, pred))
    target_names = le.classes_
    report = classification_report(test, pred, target_names=le.classes_, zero_division=0)

    """   # Plot confusion matrix and save it to a png
    cm_normalized = np.array(confusion_matrix(test, pred, normalize='true')) # normalize = 'true' or 'pred'
    confusion = pd.DataFrame(cm_normalized, index=le.classes_, columns=le.classes_ + ' (pred)')
    sns.heatmap(confusion, annot=True, cmap="Blues", fmt='.2f')
    plt.title(f'Confusion matrix')
    plt.savefig(f'/home/magali.egger/workspace/TBproject/Travail_Bachelor/ClassificationSimple/easy_ensemble_10result/confusion_matrix_{seed}.png')
    """
    # write the result in file 
    file_name = 'testset_'+str(seed)+'.csv'
    with open('/home/magali.egger/workspace/TBproject/Travail_Bachelor/ClassificationSimple/easy_ensemble_10result/'+ file_name, 'w') as f:
        f.write(str(seed) + '\n' + str(cm) +  '\n' + str(report) + '\n' + str(mice_test) + '\n\n')