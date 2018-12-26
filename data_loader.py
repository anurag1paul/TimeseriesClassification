import numpy as np
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

    def get_ids_goals(self, df):
        """
        Extraxt ids of different timeseries samples and their respective goals which will serve as target classes
        :param df: dataframe
        :return: timeseries ids, target classes
        """
        data = df[["id", "goal"]].drop_duplicates().reset_index()
        X = data["id"].values
        data.set_index("id", inplace=True)
        y = data["goal"]
        return X, y

    def get_train_test_df(self):
        return (self.df_train, self.y_train,
                self.df_test, self.y_test)

    def get_train_test_df_norm(self):
        return (self.df_train_norm, self.y_train,
                self.df_test_norm, self.y_test)
