from NN import NN
from scipy.io import loadmat
import numpy as np
import random
import matplotlib.pyplot as plt
from datetime import datetime


def get_random_n_points(n):
    """
    return an array of n random numbers.
    """
    sel = random.sample(xrange(60000), n)
    return sel

def NN_n_points(training_data, training_labels, test_data, test_labels, n):
    """
    randomly select n points from trainning set to run 1-NN classification.
    Return the error rate.
    """
    sel_points = get_random_n_points(n)
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
        error_matrix = []
        for num_point in num_points:
            error_rates.append(NN_n_points(training_data, training_labels, test_data, test_labels, num_point))
        if total_error_rates is None:
            total_error_rates = np.array(error_rates)
        else:
            total_error_rates += np.array(error_rates)
        error_matrix.append(error_rates)
    # mean
    mean_error_rates = total_error_rates / num_times
    # standard deviation
    error_matrix = np.matrix(error_matrix)
    deviation_matrix = np.square(error_matrix - mean_error_rates)
    deviation_matrix = np.sqrt(np.sum(deviation_matrix, axis=0) / float(num_times))
    deviations = np.array(deviation_matrix).flatten()
    return mean_error_rates, deviations

def plot_learning_curve(num_points, error_rates, deviations):
    plt.errorbar(num_points, error_rates, yerr=deviations, fmt='o', linestyle='-')
    plt.xlim([0, 10000])
    plt.xlabel('n')
    plt.ylabel('error rate')
    plt.title('learning curve')
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
    num_times = 1
    error_rates, deviations = NN_n_times(training_data, training_labels, test_data, test_labels, num_times, num_points)
    print 'error_rates: ', error_rates
    end = datetime.now()
    used_time = (end - start).seconds
    print 'used time: %d' % used_time
    # plot
    plot_learning_curve(num_points, error_rates, deviations)

if __name__ == '__main__':
    main()
