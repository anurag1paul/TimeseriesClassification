import pickle
from abc import ABC, abstractmethod
from subprocess import call

import numpy as np
import pandas as pd
from keras import Sequential
from keras.layers import ConvLSTM2D, Dropout, Flatten, Dense, LSTM, TimeDistributed, Conv1D, MaxPooling1D, warnings, \
    BatchNormalization, Activation, regularizers
from keras.utils import to_categorical
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from tsfresh import extract_features
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
    def _predict(self, testX):
        """
        Predict using the best model
        :param testX: test datset
        :return: predicted values
        """
        pass

    def predict(self, testx=None):
        """
        Predict using the best_model after training one model
        :return: predicted values
        """
        if self.best_model is None:
            raise Exception("Train a model first")

        if testx is None:
            testx = self.test_X

        return self._predict(testx)

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
        best_score = max(scores)
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
        return confusion_matrix(self.test_y, self.predict())


# Model using Hand-crafted features and traditional ML models
class FeatureEngineeredModel(BaseModel):

    def __init__(self, train_X, train_y, test_X, test_y, train_ids, test_ids):
        super().__init__(test_X, test_y)
        self.train_y = train_y

        self.extraction_settings = ComprehensiveFCParameters()
        X = self.generate_features(pd.concat([train_X, test_X]))

        new_train_X = X.loc[train_ids]
        new_test_X = X.loc[test_ids]

        relevant_features = self.select_features(new_train_X, self.train_y)
        print("Selected Features: {}/{}".format(len(relevant_features), X.shape[1]))

        if len(relevant_features) > 10:
            self.train_X = new_train_X[relevant_features]
            self.test_X = new_test_X[relevant_features]
        else:
            self.train_X = new_train_X
            self.test_X = new_test_X

        self.n_estimators = 40
        self.model_names = ["Random Forest", "XGBoost"]

    def generate_features(self, df):
        """
        Using tsfresh library to generate all possible features from time series
        :param df: dataframe
        :return: dataframe with generated features
        """
        df = df.reset_index()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return extract_features(df, column_id="id", impute_function=impute,
                                default_fc_parameters=self.extraction_settings)

    def select_features(self, X, y):
        """
        select the best features for the model
        :return: best features
        """
        # remove features that are constant
        X = X.loc[:, (X != X.iloc[0]).any()]
        data = {"X": X, "y": y}
        with open("data.pkl", "wb") as data_file:
            pickle.dump(data, data_file)

        call(["python3", "select_features.py"])
        with open("rel_features.pkl", "rb") as rel_features:
            relevant_features = pickle.load(rel_features)

        return list(relevant_features)

    def _predict(self, testX):
        return self.best_model.predict(testX)

    def evaluate(self):
        num_estimators = [10, 20, 30, 40, 50, 100, 150, 250]

        for i, algo in enumerate([RandomForestClassifier, XGBClassifier]):

            print("Training {}". format(self.model_names[i]))
            models = []
            scores = []

            for n_estimators in num_estimators:
                self.n_estimators = n_estimators
                model = algo(n_estimators=self.n_estimators).fit(self.train_X, self.train_y)
                score = model.score(self.test_X, self.test_y) * 100.0
                models.append(model)
                scores.append(score)

            best_model, best_score, mu, sigma = self.summarize_results(models, np.array(scores))
            print("Accuracy: {}% with n_estimators={}".format(best_score, best_model.n_estimators))
            self.models.append(best_model)
            self.scores.append(best_score)
            if best_score > self.best_score:
                self.best_model = best_model
                self.best_score = best_score

        self.n_estimators = self.best_model.n_estimators
        return self.best_model, self.best_score


# Deep Learning Models using LSTM, CNN-1D-LSTM, CNN-2D-LSTM
class DeepLearningModel(BaseModel):

    def __init__(self, train_X, train_y, test_X, test_y, train_ids, test_ids, debug=False):
        super().__init__(test_X, test_y)
        self.n_features = len(train_X.columns)
        self.n_sample_rows = len(train_X.loc[train_ids[0]])

        self.train_X = self.reformat_data(train_X, train_ids)
        self.test_X = self.reformat_data(test_X, test_ids)

        self.train_y = to_categorical(train_y)
        self.test_y_cat = to_categorical(test_y)

        # define model
        self.model_names = ["LSTM", "CNN1D-LSTM", "CNN2D-LSTM"]
        self.best_model_name = None
        self.verbose, self.epochs, self.batch_size = 0, 30, 16
        self.debug=debug

        # for CNN models
        self.n_steps, self.n_length = 1, 89

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
        model.add(LSTM(150, kernel_regularizer=regularizers.l2(0.01), input_shape=(n_timesteps, n_features)))
        model.add(Dropout(0.5))
        model.add(Dense(n_outputs, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        # fit network
        model.fit(trainX, trainy, epochs=self.epochs, batch_size=self.batch_size, verbose=self.verbose)
        # evaluate model
        _, accuracy = model.evaluate(testX, testy, batch_size=self.batch_size, verbose=0)
        if self.debug:
            _, train_accuracy = model.evaluate(trainX, trainy, batch_size=self.batch_size, verbose=0)
            print("Train Acc:{} Test Acc:{}".format(train_accuracy, accuracy))
        return model, accuracy

    def cnn1d_lstm_model(self, trainX, trainy, testX, testy):

        n_timesteps, n_outputs = trainX.shape[1], trainy.shape[1]

        # reshape data into time steps of sub-sequences
        trainX = trainX.reshape((trainX.shape[0], self.n_steps, self.n_length, self.n_features))
        testX = testX.reshape((testX.shape[0], self.n_steps, self.n_length, self.n_features))

        # define model
        model = Sequential()
        model.add(TimeDistributed(Conv1D(filters=24, kernel_size=3, kernel_regularizer=regularizers.l2(0.01)),
                                  input_shape=(None, self.n_length, self.n_features)))
        model.add(TimeDistributed(BatchNormalization()))
        model.add(TimeDistributed(Activation("relu")))
        model.add(TimeDistributed(MaxPooling1D(pool_size=2)))
        model.add(TimeDistributed(Flatten()))
        model.add(LSTM(100, kernel_regularizer=regularizers.l2(0.01)))
        model.add(Dropout(0.5))
        model.add(Dense(n_outputs, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        # fit network
        model.fit(trainX, trainy, epochs=self.epochs, batch_size=self.batch_size, verbose=self.verbose)
        # evaluate model
        _, accuracy = model.evaluate(testX, testy, batch_size=self.batch_size, verbose=0)
        if self.debug:
            _, train_accuracy = model.evaluate(trainX, trainy, batch_size=self.batch_size, verbose=0)
            print("Train Acc:{} Test Acc:{}".format(train_accuracy, accuracy))
        return model, accuracy

    def cnn2d_lstm_model(self, trainX, trainy, testX, testy):

        n_timesteps, n_outputs = trainX.shape[1], trainy.shape[1]

        # reshape into subsequences (samples, time steps, rows, cols, channels)
        trainX = trainX.reshape((trainX.shape[0], self.n_steps, 1, self.n_length, self.n_features))
        testX = testX.reshape((testX.shape[0], self.n_steps, 1, self.n_length, self.n_features))

        # define model
        model = Sequential()
        model.add(ConvLSTM2D(filters=20, kernel_size=(1,3), kernel_regularizer=regularizers.l2(0.01),
                             input_shape=(self.n_steps, 1, self.n_length, self.n_features)))
        model.add(BatchNormalization())
        model.add(Activation("relu"))
        model.add(Dropout(0.5))
        model.add(Flatten())
        model.add(Dense(n_outputs, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        # fit network
        model.fit(trainX, trainy, epochs=self.epochs, batch_size=self.batch_size, verbose=self.verbose)
        # evaluate model
        _, accuracy = model.evaluate(testX, testy, batch_size=self.batch_size, verbose=0)
        if self.debug:
            _, train_accuracy = model.evaluate(trainX, trainy, batch_size=self.batch_size, verbose=0)
            print("Train Acc:{} Test Acc:{}".format(train_accuracy, accuracy))
        return model, accuracy

    def _predict(self, testX):

        if self.best_model_name == "CNN1D-LSTM":
            testX = testX.reshape((testX.shape[0], self.n_steps, self.n_length, self.n_features))

        elif self.best_model_name == "CNN2D-LSTM":
            testX = testX.reshape((testX.shape[0], self.n_steps, 1, self.n_length, self.n_features))

        y_pred = [np.argmax(l) for l in self.best_model.predict(testX, batch_size=self.batch_size, verbose=0)]

        return y_pred

    def evaluate(self):
        n_reps = 5

        for i, algo in enumerate([self.simple_lstm_model, self.cnn1d_lstm_model, self.cnn2d_lstm_model]):

            print("Training {}". format(self.model_names[i]))
            models = []
            scores = []

            for r in range(n_reps):
                model, score = algo(self.train_X, self.train_y, self.test_X, self.test_y_cat)
                models.append(model)
                scores.append(score * 100)

            best_model, best_score, mu, sigma = self.summarize_results(models, np.array(scores))
            print("Accuracy: Max:{}% Avg:{}% (+/-{})".format(best_score, round(mu, 3), round(sigma, 3)))
            self.models.append(best_model)
            self.scores.append(best_score)
            if best_score > self.best_score:
                self.best_model = best_model
                self.best_score = best_score
                self.best_model_name = self.model_names[i]

        return self.best_model, self.best_score
