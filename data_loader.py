import numpy as np
from keras.utils import to_categorical
from pandas import read_csv


class DataLoader:

    def __init__(self, train_file, test_file):

        df_train = read_csv(train_file)
        df_test = read_csv(test_file)

        self.df_train = df_train.set_index("id").drop(["goal"], axis=1)
        self.df_test = df_test.set_index("id").drop(["goal"], axis=1)

        self.train_ids, self.y_train = self.get_ids_goals(df_train)
        self.test_ids, self.y_test = self.get_ids_goals(df_test)

        self.df_mean = np.mean(self.df_train)
        self.df_std = np.std(self.df_train) + 1e-6

        self.df_train_norm = (self.df_train - self.df_mean) / self.df_std
        self.df_test_norm = (self.df_test - self.df_mean) / self.df_std

        self.X_train = self.reformat_data(self.df_train_norm, self.train_ids)
        self.X_test = self.reformat_data(self.df_test_norm, self.test_ids)

        self.num_features = len(self.df_train.columns)
        self.num_sample_rows = len(self.df_train.loc[self.train_ids[0]])

    def get_ids_goals(self, df):
        """
        Extraxt ids of different timeseries samples and their respective goals which will serve as target classes
        :param df: dataframe
        :return: timeseries ids, target classes
        """
        data = df[["id", "goal"]].drop_duplicates().reset_index()
        X = data["id"].values
        y = data["goal"].values
        return X, y

    def reformat_data(self, df, ids):
        """
        Converts the data to the format needed by Keras
        :param df: data frame
        :param ids: unique timeseries ids
        :return: reformatted data
        """

        data = np.zeros((len(ids), self.num_sample_rows, self.num_features))
        idx = 0
        for i in ids:
            sample = df.loc[i]
            data[idx] = sample.values
            idx += 1
        return data

    def get_train_test_norm(self):
        return (self.X_train, to_categorical(self.y_train),
                self.X_test, to_categorical(self.y_test))

    def get_train_test_df(self):
        return (self.df_train, self.y_train,
                self.df_test, self.y_test)

    def get_train_test_df_norm(self):
        return (self.df_train_norm, self.y_train,
                self.df_test_norm, self.y_test)
