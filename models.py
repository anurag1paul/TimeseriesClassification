import warnings
from abc import ABC, abstractmethod

import numpy as np
from keras import Sequential
from keras.layers import ConvLSTM2D, Dropout, Flatten, Dense, LSTM, TimeDistributed, Conv1D, MaxPooling1D
from keras.utils import to_categorical
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from tsfresh import extract_features, select_features
from tsfresh.feature_extraction import ComprehensiveFCParameters
from tsfresh.utilities.dataframe_functions import impute
from xgboost import XGBClassifier


class BaseModel(ABC):

    def __init__(self, test_X, test_y):
        self.test_X = test_X
        self.test_y = test_y
        self.models = []
        self.scores = []
        self.best_model = None
        self.best_score = 0

    @abstractmethod
    def _predict(self, model, testX, testy):
        """
        Predict using the provided model
        :param model: trained ML model
        :param testX: test datset
        :param testy: test target values
        :return: predicted values, score
        """
        pass

    def predict(self, testx=None, testy=None):
        """
        Predict using the best_model after training one model
        :return: predicted values, score
        """
        if self.best_model is None:
            raise Exception("Train a model first")

        if testx is None or testy is None:
            testx = self.test_X
            testy = self.test_y

        return self._predict(self.best_model, testx, testy)

    @abstractmethod
    def evaluate(self):
        """
        Train model multiple times and select the best model
        :return: best_model, best_score, mean and std deviation
        """
        pass

    def summarize_results(self, models, scores):
        """
        Calculate and return the best model, score and the performance statistics
        :return: best_model, best_model's score, mean_score, standard deviation of the scores
        """
        mu = np.mean(scores)
        sigma = np.std(scores)
        best_model = models[np.argmax(scores)]
        best_score = np.max(self.scores)
        return best_model, best_score, mu, sigma

    def get_best_model(self):
        """
        returns None if best models is not trained
        :return:
        """
        return self.best_model

    def get_confusion_matrix(self):
        """
        Use Sklearn's function to generate confusion matrix for the test data
        :return: confusion matrix
        """
        y_pred, _ = self.predict()
        return confusion_matrix(self.test_y, y_pred)


# Model using Hand-crafted features and traditional ML models
class FeatureEngineeredModel(BaseModel):

    def __init__(self, train_X, train_y, test_X, test_y):
        super().__init__(test_X, test_y)
        self.extraction_settings = ComprehensiveFCParameters()
        new_train_X = self.generate_features(train_X)
        new_test_X = self.generate_features(test_X)
        self.train_y = train_y
        relevant_features = self.select_features(new_train_X, self.train_y)
        self.train_X = new_train_X[relevant_features]
        self.test_X = new_test_X[relevant_features]
        self.n_estimators = 40

        self.model_names = ["Random Forest", "XGBoost"]

    def generate_features(self, df):
        """
        Using tsfresh library to generate all possible features from time series
        :param df: dataframe
        :return: dataframe with generated features
        """
        df = df.reset_index()
        return extract_features(df, column_id="id", impute_function=impute,
                                default_fc_parameters=self.extraction_settings)

    def select_features(self, X, y):
        """
        select the best features for the model
        :return: best features
        """
        # remove features that are constant
        X = X.loc[:, (X != X.iloc[0]).any()]
        relevant_features = set()

        for label in y.unique():
            y_binary = np.array(y == label)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                X_filtered = select_features(X, y_binary)
                print("Number of relevant features for class {}: {}/{}".format(label, X_filtered.shape[1], X.shape[1]))
                relevant_features = relevant_features.union(set(X_filtered.columns))

        return list(relevant_features)

    def _predict(self, model, testX, testy):
        pred_y = model.predict(testX)
        return pred_y, model.score(testy, pred_y)

    def evaluate(self):
        models = []
        scores = []
        num_estimators = [10, 20, 30, 40, 50, 100, 150, 250]

        for algo in [RandomForestClassifier, XGBClassifier]:

            for n_estimators in num_estimators:
                self.n_estimators = n_estimators
                model = algo(n_estimators=self.n_estimators).fit(self.train_X, self.train_y)
                score = model.score(self.test_X, self.test_y) * 100.0
                models.append(model)
                scores.append(score)

            best_model, best_score, mu, sigma = self.summarize_results(models, np.array(scores))
            print("Accuracy: {}% (+/-{}) (95% confidence interval)".format(round(mu, 3), 2 * round(sigma, 3)))
            self.models.append(best_model)
            self.scores.append(best_score)
            if best_score > self.best_score:
                self.best_model = best_model
                self.best_score = best_score

        self.n_estimators = self.best_model.n_estimators
        return self.best_model, self.best_score, mu, sigma


# Deep Learning Models using LSTM, CNN-1D-LSTM, CNN-2D-LSTM
class DeepLearningModel(BaseModel):

    def __init__(self, train_X, train_y, test_X, test_y, train_ids, test_ids):
        super().__init__(test_X, test_y)
        self.n_features = len(self.train_X.columns)
        self.n_sample_rows = len(self.train_X.loc[train_ids[0]])

        self.train_X = self.reformat_data(train_X, train_ids)
        self.test_X = self.reformat_data(test_X, test_ids)

        self.train_y = to_categorical(train_y)
        self.train_X = to_categorical(test_y)

        # define model
        self.model_names = ["LSTM", "CNN1D-LSTM", "CNN2D-LSTM"]
        self.verbose, self.epochs, self.batch_size = 0, 25, 16

    def reformat_data(self, df, ids):
        """
        Converts the data to the format needed by Keras
        :param df: data frame
        :param ids: unique timeseries ids
        :return: reformatted data
        """
        data = np.zeros((len(ids), self.n_sample_rows, self.n_features))
        idx = 0
        for i in ids:
            sample = df.loc[i]
            data[idx] = sample.values
            idx += 1
        return data

    def simple_lstm_model(self, trainX, trainy, testX, testy):

        n_timesteps, n_features, n_outputs = trainX.shape[1], trainX.shape[2], trainy.shape[1]
        model = Sequential()
        model.add(LSTM(100, input_shape=(n_timesteps,n_features)))
        model.add(Dropout(0.5))
        model.add(Dense(100, activation='relu'))
        model.add(Dense(n_outputs, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        # fit network
        model.fit(trainX, trainy, epochs=self.epochs, batch_size=self.batch_size, verbose=self.verbose)
        # evaluate model
        _, accuracy = model.evaluate(testX, testy, batch_size=self.batch_size, verbose=0)
        return model, accuracy

    def cnn1d_lstm_model(self, trainX, trainy, testX, testy):

        n_timesteps, n_features, n_outputs = trainX.shape[1], trainX.shape[2], trainy.shape[1]

        # reshape data into time steps of sub-sequences
        n_steps, n_length = 1, 89
        trainX = trainX.reshape((trainX.shape[0], n_steps, n_length, self.n_features))
        testX = testX.reshape((testX.shape[0], n_steps, n_length, self.n_features))

        # define model
        model = Sequential()
        model.add(TimeDistributed(Conv1D(filters=32, kernel_size=3, activation='relu'), input_shape=(None,n_length,n_features)))
        model.add(TimeDistributed(Conv1D(filters=64, kernel_size=3, activation='relu')))
        model.add(TimeDistributed(MaxPooling1D(pool_size=2)))
        model.add(TimeDistributed(Flatten()))
        model.add(LSTM(100))
        model.add(Dropout(0.5))
        model.add(Dense(100, activation='relu'))
        model.add(Dense(n_outputs, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        # fit network
        model.fit(trainX, trainy, epochs=self.epochs, batch_size=self.batch_size, verbose=self.verbose)
        # evaluate model
        _, accuracy = model.evaluate(testX, testy, batch_size=self.batch_size, verbose=0)
        return model, accuracy

    def cnn2d_lstm_model(self, trainX, trainy, testX, testy):

        n_timesteps, n_outputs = trainX.shape[1], trainy.shape[1]

        # reshape into subsequences (samples, time steps, rows, cols, channels)
        n_steps, n_length = 1, 89
        trainX = trainX.reshape((trainX.shape[0], n_steps, 1, n_length, self.n_features))
        testX = testX.reshape((testX.shape[0], n_steps, 1, n_length, self.n_features))

        # define model
        model = Sequential()
        model.add(ConvLSTM2D(filters=128, kernel_size=(1,3), activation='relu',
                             input_shape=(n_steps, 1, n_length, self.n_features)))
        model.add(Dropout(0.5))
        model.add(Flatten())
        model.add(Dense(100, activation='relu'))
        model.add(Dense(n_outputs, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        # fit network
        model.fit(trainX, trainy, epochs=self.epochs, batch_size=self.batch_size, verbose=self.verbose)
        # evaluate model
        _, accuracy = model.evaluate(testX, testy, batch_size=self.batch_size, verbose=0)
        return accuracy

    def _predict(self, model, testX, testy):
        pred_y = model.predict(testX, batch_size=self.batch_size, verbose=0)
        _, accuracy = model.evaluate(testX, testy, batch_size=self.batch_size, verbose=0)
        return pred_y, accuracy

    def evaluate(self):
        models = []
        scores = []

        for algo in [self.simple_lstm_model, self.cnn1d_lstm_model, self. cnn2d_lstm_model]:

            for r in range(10):
                model, score = algo(self.train_X, self.train_y, self.test_X, self.test_y)
                models.append(model)
                scores.append(score)

            best_model, best_score, mu, sigma = self.summarize_results(models, np.array(scores))
            print("Accuracy: {}% (+/-{}) (95% confidence interval)".format(round(mu, 3), 2 * round(sigma, 3)))
            self.models.append(best_model)
            self.scores.append(best_score)
            if best_score > self.best_score:
                self.best_model = best_model
                self.best_score = best_score

        return self.best_model, self.best_score, mu, sigma
