import os

from pandas import read_excel
from sklearn.model_selection import train_test_split


def prepare_data(excel_file, train_file, test_file, test_size=0.2, gen=True):
    """
    Loads the given dataset, renames columns and writes train and test datasets
    :param excel_file: name of the raw data set
    :param train_file: name of the train dataset
    :param test_file: name of the test dataset
    :param test_size: size of the test set
    :return: data containing unique ids and goals
    """
    df = read_excel(excel_file)
    df.rename(columns={"ID_TestSet": "id"}, inplace=True)
    data = df[["id", "goal"]].drop_duplicates().reset_index()

    if not (os.path.isfile(train_file) and os.path.isfile(test_file)) and gen:
        X = data["id"]
        y = data["goal"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)

        df2 = df.set_index("id").drop(["file"], axis=1)

        df_train = df2.loc[X_train,:]
        df_test = df2.loc[X_test,:]
        df_train.to_csv(train_file)
        df_test.to_csv(test_file)

    return data
