import itertools

import numpy as np
from matplotlib import pyplot as plt


def plot_confusion_matrix(confusion_matrix, classes,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function plots the confusion matrix.
    """
    plt.imshow(confusion_matrix, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)

    thresh = confusion_matrix.max() / 2.
    for i, j in itertools.product(range(confusion_matrix.shape[0]), range(confusion_matrix.shape[1])):
        plt.text(j, i, format(confusion_matrix[i, j], 'd'),
                 horizontalalignment="center",
                 color="white" if confusion_matrix[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()


def plot_class_distribution(classes):
    """
    This function plots the class distribution in the data
    :param classes: series object
    :return: None
    """
    plt.figure()
    plt.bar(classes.index, classes)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes.index)
    plt.title("Class Distribution")
    plt.xlabel("Classes")
    plt.ylabel("Number of timeseries samples")
    plt.show()


def plot_best_accuracies(model_names, accuracies):
    """
    Plots a bar plot of model accuracies
    :param model_names: list of model names
    :param accuracies: list of best accuracies
    :return: None
    """
    plt.rcdefaults()
    fig, ax = plt.subplots()
    y_pos = np.arange(len(model_names))


    ax.barh(y_pos, accuracies, align='center',
            color='blue')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(model_names)
    ax.invert_yaxis()
    ax.set_xlabel('Best Accuracy')
    ax.set_title('Best Accuracy for all classifiers')
    plt.xlim((0,100))

    plt.show()
