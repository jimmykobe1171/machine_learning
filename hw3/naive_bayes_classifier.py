import numpy as np

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