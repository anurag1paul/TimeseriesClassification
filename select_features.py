import pickle
import warnings

import numpy
from tsfresh import select_features


def relevant_features(X, y):
    relevant_features = set()
    for label in y.unique():
        y_binary = numpy.array(y == label)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            X_filtered = select_features(X, y_binary)
            print("Number of relevant features for class {}: {}/{}".format(label, X_filtered.shape[1], X.shape[1]))
            relevant_features = relevant_features.union(set(X_filtered.columns))
    return relevant_features


if __name__ == "__main__":
    with open("data.pkl", "rb") as data_file:
        data = pickle.load(data_file)
    rel_features = relevant_features(data["X"], data["y"])
    with open("rel_features.pkl", "wb") as features_file:
        pickle.dump(rel_features, features_file)
