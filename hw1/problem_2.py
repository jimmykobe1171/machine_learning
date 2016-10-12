from NN import NN
from scipy.io import loadmat
import numpy as np
import random
import matplotlib.pyplot as plt
from datetime import datetime
from NN import EuclideanDistance


def get_sorted_indexes_from_central_point(label, traning_data, training_labels):
    """
    calculate the central point of given label, and then calculate
    the sorted indexes of training data based on central point.
    """
    indexes = np.argwhere(training_labels.flatten()==label)
    indexes = indexes.flatten()
    sel_data = traning_data[indexes]
    central_point = np.matrix(np.sum(sel_data, axis=0) / float(sel_data.shape[0]))
    # calculate distances to central point
    distance_matrix = EuclideanDistance.calculate(sel_data, central_point)
    order = np.argsort(np.array(distance_matrix).flatten())
    sorted_indexes = indexes[order]
    return sorted_indexes

def get_prototype_selection_indexes(num_point, traning_data, training_labels):
    indexes = []
    # 10 labels
    for i in range(10):
        sorted_indexes = get_sorted_indexes_from_central_point(i, traning_data, training_labels)
        indexes.append(sorted_indexes)

    sel_indexes = [index[:num_point/10] for index in indexes]
    sel_indexes = np.concatenate(sel_indexes)
    return sel_indexes


def NN_n_points(training_data, training_labels, test_data, test_labels, n):
    """
    randomly select n points from trainning set to run 1-NN classification.
    Return the error rate.
    """
    sel_points = get_prototype_selection_indexes(n, training_data, training_labels)
    sel_training_data = training_data[sel_points]
    sel_training_labels = training_labels[sel_points]
    error_rate = NN.get_instance().classify(sel_training_data, sel_training_labels, test_data, test_labels)
    return error_rate

def NN_n_times(training_data, training_labels, test_data, test_labels, num_times, num_points):
    """
    run 1-NN n times independently. Return the error rate.
    """
    total_error_rates = None
    for i in range(num_times):
        error_rates = []
        for num_point in num_points:
            error_rates.append(NN_n_points(training_data, training_labels, test_data, test_labels, num_point))
        if total_error_rates is None:
            total_error_rates = np.array(error_rates)
        else:
            total_error_rates += np.array(error_rates)

    total_error_rates = total_error_rates / float(num_times)
    return total_error_rates

def plot_learning_curve(num_points, error_rates):
    plt.plot(num_points, error_rates)
    plt.show()

def main():
    start = datetime.now()
    # read data
    ocr = loadmat('ocr.mat')
    training_data = ocr.get('data')
    training_data = training_data.astype('float')
    training_labels = ocr.get('labels')
    test_data = ocr.get('testdata')
    test_data = test_data.astype('float')
    test_labels = ocr.get('testlabels')

    # classification
    num_points = [1000, 2000, 4000, 8000]
    # num_points = [1000]
    num_times = 1
    error_rates = NN_n_times(training_data, training_labels, test_data, test_labels, num_times, num_points)
    print 'error_rates: ', error_rates
    end = datetime.now()
    used_time = (end - start).seconds
    print 'used time: %d' % used_time
    # plot
    # plot_learning_curve(num_points, error_rates)

if __name__ == '__main__':
    main()
