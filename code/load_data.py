import os
from sklearn.datasets import load_svmlight_files
import numpy as np
from sklearn.model_selection import train_test_split


def load_files(files_to_load, num_of_features, change_range=False):
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = 'dataset/url_svmlight'

    files = map(lambda x: os.path.join(root_dir, data_dir, x), os.listdir(os.path.join(root_dir, data_dir))[: files_to_load])

    return_array = load_svmlight_files(files)
    features = []
    labels = []

    for each_value in return_array:
        if type(each_value) != np.ndarray:
            features.extend(each_value[:, :num_of_features].toarray().tolist())
        else:
            labels.extend(each_value.tolist())

    features = np.array(features)
    labels = np.array(labels)

    print(features.shape)
    print(np.unique(labels, return_counts=True))

    if change_range:
        labels = np.array(list(map(lambda x: int(x>0), labels)))
        print(np.unique(labels, return_counts=True))
        labels = labels.reshape((-1, 1))

    trainx, testx, trainy, testy = train_test_split(features, labels, train_size=0.80)

    return trainx, trainy, testx, testy


if __name__ == '__main__':
    load_files(1, 10, True)
