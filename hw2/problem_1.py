from scipy.io import loadmat
import numpy as np
from datetime import datetime


class NaiveBayesClassifier(object):
    def __init__(self, training_data, training_labels, labels):
        """
        train and save model parameters pis and mus.
        labels is an array of available class labels, for binary problem,
        it will be [0, 1].
        """
        mus = []
        pis = []
        for i in labels:
            indexes = np.where(training_labels==i)[0]
            data = training_data[indexes]
            row_num = data.shape[0]
            sum_data = np.sum(data, axis=0)
            mu = (sum_data + 1) / float(row_num + 2)
            mu = np.array(mu).flatten()
            pi = float(row_num) / training_data.shape[0]
            mus.append(mu)
            pis.append(pi)

        self.mus = np.array(mus)
        self.pis = np.array(pis)
        self.labels = np.array(labels)

    def classify(self, test_data, test_labels):
        """
        classify test data using the trained model,
        return error_rate.
        """
        pis_matrix = np.tile(self.pis, (test_data.shape[0], 1))
        result_matrix = np.log(pis_matrix) + test_data * np.log(self.mus.transpose()) + \
                        (np.ones(test_data.shape)-test_data) * np.log(np.ones(self.mus.transpose().shape)-self.mus.transpose())
        max_indexes = np.argmax(result_matrix, axis=1)
        classified_labels = self.labels[np.array(max_indexes).flatten()]
        error_num = (classified_labels - np.array(test_labels).flatten()).nonzero()[0].shape[0]
        error_rate = float(error_num) / test_labels.shape[0]
        return error_rate


def get_binary_classification_training_set(training_data, training_labels):
    """
    generate the training data and training labels for problem c
    """
    # {1,16,20} is 0 class, {17,18,19} is 1 class
    negative_indexes = np.where(np.in1d(training_labels.flatten(), [1,16,20]))[0]
    positive_indexes = np.where(np.in1d(training_labels.flatten(), [17,18,19]))[0]
    total_indexes = np.concatenate((negative_indexes, positive_indexes))
    new_labels = np.concatenate((np.zeros((negative_indexes.shape[0], 1)), np.ones((positive_indexes.shape[0], 1))))
    return training_data[total_indexes], new_labels

def get_binary_classification_test_set(test_data, test_labels):
    """
    generate the test data and test labels for problem c
    """
    # {1,16,20} is 0 class, {17,18,19} is 1 class
    negative_indexes = np.where(np.in1d(test_labels.flatten(), [1,16,20]))[0]
    positive_indexes = np.where(np.in1d(test_labels.flatten(), [17,18,19]))[0]
    total_indexes = np.concatenate((negative_indexes, positive_indexes))
    new_labels = np.concatenate((np.zeros((negative_indexes.shape[0], 1)), np.ones((positive_indexes.shape[0], 1))))
    return test_data[total_indexes], new_labels

def problem_b(training_data, training_labels, test_data, test_labels):
    """
    problem b solution.
    """
    labels = [i for i in range(1, 21)]
    classifier = NaiveBayesClassifier(training_data, training_labels, labels)
    training_error_rate = classifier.classify(training_data, training_labels)
    test_error_rate = classifier.classify(test_data, test_labels)
    print 'training_error_rate: ', training_error_rate
    print 'test_error_rate: ', test_error_rate

def problem_c(training_data, training_labels, test_data, test_labels):
    """
    problem c solution,
    return the classifier for the use of problem d
    """
    labels = [0, 1]
    binary_training_data, binary_training_labels = get_binary_classification_training_set(training_data, training_labels)
    binary_test_data, binary_test_labels = get_binary_classification_test_set(test_data, test_labels)
    binary_classifier = NaiveBayesClassifier(binary_training_data, binary_training_labels, labels)
    binary_training_error_rate = binary_classifier.classify(binary_training_data, binary_training_labels)
    binary_test_error_rate = binary_classifier.classify(binary_test_data, binary_test_labels)
    print 'binary_training_error_rate: ', binary_training_error_rate
    print 'binary_test_error_rate: ', binary_test_error_rate
    return binary_classifier

def problem_d(classifier):
    pis, mus = classifier.pis, classifier.mus
    # calculate alpha_0
    one_minus_mus = np.ones(mus.shape) - mus
    # alpha_0 = np.divide(pis[1], pis[0]) + np.sum(np.log(np.divide(one_minus_mus[1], one_minus_mus[0])))
    # calculate alpha_j
    alpha_js = np.log(np.multiply(np.divide(one_minus_mus[0], one_minus_mus[1]), np.divide(mus[1], mus[0])))
    sorted_indexes = np.argsort(alpha_js)
    smallest_20_indexes, largest_20_indexes = sorted_indexes[:20], sorted_indexes[-20:]
    # read vocab
    with open('news.vocab', 'r') as f:
        vocab = [word.strip() for word in f]

    smallest_20 = [vocab[i] for i in smallest_20_indexes]
    largest_20 = [vocab[i] for i in largest_20_indexes]
    largest_20.reverse()
    print smallest_20
    print largest_20

def main():
    start = datetime.now()
    # read data
    news = loadmat('news.mat')
    training_data = news.get('data')
    training_data = training_data.astype('float')
    training_labels = news.get('labels')
    test_data = news.get('testdata')
    test_data = test_data.astype('float')
    test_labels = news.get('testlabels')

    # problem (b) classification
    problem_b(training_data, training_labels, test_data, test_labels)

    # problem (c) classification
    classifier = problem_c(training_data, training_labels, test_data, test_labels)

    # problem (d)
    problem_d(classifier)

    # test case
    # training_data = np.matrix([[1, 0, 0], [1, 1, 0], [0, 1, 1], [0, 0, 1], [0, 1, 0]])
    # training_labels = np.matrix([[0], [0], [0], [1], [1]])
    # classifier = NaiveBayesClassifier(training_data, training_labels, [0, 1])
    # test_error_rate = classifier.classify(training_data, training_labels)
    # print test_error_rate
    
    end = datetime.now()
    used_time = (end - start).seconds
    print 'used time: %d' % used_time

if __name__ == '__main__':
    main()