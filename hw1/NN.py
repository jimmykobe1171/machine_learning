import numpy as np


class EuclideanDistance(object):
    @staticmethod
    def calculate(matrix_a, matrix_b):
        """
        calculate the Euclidean distance of two matrix, distance = A^2 + B^2 - 2A*transpose(B)
        """
        # calculate the dot production of two matrix
        dot_prod = np.dot(matrix_a, matrix_b.transpose())

        # calculate square matrix of matrix_a
        square_matrix_a = np.matrix(np.sum(np.square(matrix_a), axis=1))
        square_matrix_a = square_matrix_a.transpose()
        # copy the column direction
        square_matrix_a = np.tile(square_matrix_a, (1, dot_prod.shape[1]))

        # calculate square matrix of matrix b
        square_matrix_b = np.matrix(np.sum(np.square(matrix_b), axis=1))
        # copy the row direction
        square_matrix_b = np.tile(square_matrix_b, (dot_prod.shape[0], 1))

        distance_matrix = square_matrix_a + square_matrix_b - dot_prod*2
        distance_matrix = np.sqrt(distance_matrix)
        # print distance_matrix
        return distance_matrix


class NN(object):
    # used for cache
    _square_test_data = None

    @classmethod
    def get_instance(cls):
        return NN()

    def classify(self, training_data, training_labels, test_data, test_labels):
        """
        1-NN classification, return error rate of the classification.
        """
        num_rows = training_data.shape[0]
        distance_matrix = EuclideanDistance.calculate(training_data, test_data)
        distance_matrix = np.array(distance_matrix)
        # an array that contains indexes of closest training point, should be 1xtest_data.rows
        indexes = np.argmin(distance_matrix, axis=0)
        classified_labels = training_labels[indexes]
        # calculate error rate
        num_error = sum([1 for i in range(num_rows) if classified_labels[i][0] != test_labels[i][0]])
        error_rate = num_error / float(num_rows)
        return error_rate
