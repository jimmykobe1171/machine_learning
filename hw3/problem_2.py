import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
from scipy import sparse
from random import shuffle
from datetime import datetime
from naive_bayes_classifier import NaiveBayesClassifier
from sklearn.model_selection import KFold


def read_training_data():
    data = pd.read_csv('reviews_tr.csv')
    return data

def read_test_data():
    data = pd.read_csv('reviews_te.csv')
    return data

def calculate_idf_matrix(unigram_feature):
    """
    generate a diagonal idf matrix to facilitate tfidf calculation.
    """
    df_matrix = np.sum(unigram_feature>0, axis=0)
    size = unigram_feature.shape[0]
    idf_matrix = np.log10(size) - np.log10(df_matrix)
    # construct diagonal matrix for multiply
    row = np.arange(unigram_feature.shape[1])
    diag = sparse.csr_matrix((np.array(idf_matrix).flatten(), (row, row)))
    return diag

def generate_unigram_feature(corpus):
    vectorizer = CountVectorizer()
    feature_matrix = vectorizer.fit_transform(corpus)
    return feature_matrix, vectorizer

def generate_tfidf_training_feature(unigram_feature):
    diag = calculate_idf_matrix(unigram_feature)
    tfidf_matrix = unigram_feature * diag
    return tfidf_matrix

def generate_tfidf_test_feature(unigram_training_feature, unigram_test_feature):
    diag = calculate_idf_matrix(unigram_training_feature)
    tfidf_matrix = unigram_test_feature * diag
    return tfidf_matrix

def generate_bigram_feature(corpus):
    vectorizer = CountVectorizer(ngram_range=(2,2))
    feature_matrix = vectorizer.fit_transform(corpus)
    return feature_matrix, vectorizer

def generate_tfidf_variant_training_feature(unigram_feature):
    diag = calculate_idf_matrix(unigram_feature)
    new_data = 1 + np.log10(unigram_feature.data)
    new_unigram_feature = sparse.csr_matrix(unigram_feature)
    new_unigram_feature.data = new_data
    tfidf_matrix = new_unigram_feature * diag
    return tfidf_matrix

def generate_tfidf_variant_test_feature(unigram_training_feature, unigram_test_feature):
    diag = calculate_idf_matrix(unigram_training_feature)
    new_data = 1 + np.log10(unigram_test_feature.data)
    new_unigram_test_feature = sparse.csr_matrix(unigram_test_feature)
    new_unigram_test_feature.data = new_data
    tfidf_matrix = new_unigram_test_feature * diag
    return tfidf_matrix


def generate_random_indexes(number):
    # randomize
    indexes = [i for i in range(number)]
    shuffle(indexes)
    return indexes

class AveragedPerceptron(object):
    def __init__(self, training_data, training_labels):
        self.w, self.b = self.train(training_data, training_labels)

    def train(self, training_data, training_labels):
        """
        training process, run 2n+1 times and return is final n+1 classifier
        """
        feature_length = training_data.shape[1]
        training_points_num = training_data.shape[0]
        # initilization
        w = np.zeros(feature_length)
        b = 0
        c = 1
        w_sum = None
        b_sum = 0
        for i in range(2):
            # randomize training data
            random_indexes = generate_random_indexes(training_points_num)
            random_training_data = training_data[random_indexes]
            random_training_labels = training_labels[random_indexes]

            for j in range(training_points_num):
                feature = random_training_data[j]
                feature = feature.toarray().flatten()
                true_label = random_training_labels[j]
                sign = true_label * (np.dot(feature, w) + b)
                if sign <= 0:
                    # update w_sum, b_sum for last n+1 classifiers
                    if (i == 0 and j == training_points_num - 1) or i == 1:
                        if w_sum is None:
                            w_sum = c*w
                        else:
                            w_sum += c*w
                        b_sum += c*b
                    # update w and b and reset c
                    # print 'sign: ', sign
                    w = w + true_label*feature
                    b = b + true_label
                    c = 1
                else:
                    c += 1
        # last classifier
        w_sum += c*w
        b_sum += c*b
        # print w_sum, b_sum
        return w_sum, b_sum

    def classify(self, test_data, test_labels):
        result_matrix = test_data * np.matrix(self.w).transpose() + self.b
        # print result_matrix, result_matrix.shape
        result_matrix[np.where(result_matrix<=0)[0]] = -1
        result_matrix[np.where(result_matrix>0)[0]] = 1
        result_matrix = np.array(result_matrix).flatten()
        correct_num = (np.where(result_matrix - test_labels == 0)[0]).shape[0]
        total_num = test_labels.shape[0]
        error_rate = (total_num - correct_num) / float(total_num)
        return error_rate


def k_fold_training(training_feature, training_labels):
    sum_error_rate = 0
    kf = KFold(n_splits=5)
    for train, test in kf.split(training_labels):
        tmp_training_feature = training_feature[train]
        tmp_training_labels = training_labels[train]
        tmp_test_feature = training_feature[test]
        tmp_test_labels = training_labels[test]

        unigram_perceptron = AveragedPerceptron(tmp_training_feature, tmp_training_labels)
        error_rate = unigram_perceptron.classify(tmp_test_feature, tmp_test_labels)
        print 'tmp error_rate: ', error_rate
        sum_error_rate += error_rate

    return sum_error_rate / 5


def test_perceptron(training_corpus, training_labels, test_corpus, test_labels):
    # set label 0 to -1
    new_training_labels = np.array(training_labels)
    new_training_labels[np.where(new_training_labels==0)[0]] = -1
    new_test_labels = np.array(test_labels)
    new_test_labels[np.where(new_test_labels==0)[0]] = -1

    # unigram 0.1117
    unigram_training_feature, vectorizer = generate_unigram_feature(training_corpus)
    # unigram_test_feature = vectorizer.transform(test_corpus)
    # unigram_perceptron = AveragedPerceptron(unigram_training_feature, new_training_labels)
    # error_rate = unigram_perceptron.classify(unigram_test_feature, new_test_labels)
    # print 'error_rate: ', error_rate
    #  k-fold error rate: 0.113415
    # error_rate = k_fold_training(unigram_training_feature, new_training_labels)
    # print 'error_rate: ', error_rate
    

    # tfidf 0.133065
    tfidf_training_feature = generate_tfidf_training_feature(unigram_training_feature)
    # tfidf_test_feature = generate_tfidf_test_feature(unigram_training_feature, unigram_test_feature)
    # tfidf_perceptron = AveragedPerceptron(tfidf_training_feature, new_training_labels)
    # error_rate = tfidf_perceptron.classify(tfidf_test_feature, new_test_labels)
    # print 'error_rate: ', error_rate
    #  k-fold error rate: 
    error_rate = k_fold_training(tfidf_training_feature, new_training_labels)
    print 'tfidf error_rate: ', error_rate

    # bigram
    # print 'generating bigram feature...'
    # bigram_training_feature, vectorizer = generate_bigram_feature(training_corpus)
    # bigram_test_feature = vectorizer.transform(test_corpus)
    # print 'building bigram perceptron...'
    # bigram_perceptron = AveragedPerceptron(bigram_training_feature, new_training_labels)
    # error_rate = bigram_perceptron.classify(bigram_test_feature, new_test_labels)
    # print 'error_rate: ', error_rate
    #  k-fold error rate: 
    # error_rate = k_fold_training(bigram_training_feature, new_training_labels)
    # print 'error_rate: ', error_rate
    
    # tfidf variant 
    tfidf_variant_training_feature = generate_tfidf_variant_training_feature(unigram_training_feature)
    # tfidf_variant_test_feature = generate_tfidf_variant_test_feature(unigram_training_feature, unigram_test_feature)
    # tfidf_variant_perceptron = AveragedPerceptron(tfidf_variant_training_feature, new_training_labels)
    # error_rate = tfidf_variant_perceptron.classify(tfidf_variant_test_feature, new_test_labels)
    # print 'error_rate: ', error_rate
    #  k-fold error rate: 
    error_rate = k_fold_training(tfidf_variant_training_feature, new_training_labels)
    print 'tfidf variant error_rate: ', error_rate

def test_naive_bayes():
    # unigram
    # labels = [-1, 1]
    # unigram_training_feature = np.where(unigram_training_feature>1)

    # covert >1 to 1
    # new_data = unigram_training_feature.data
    # new_data = unigram_training_feature.data[np.where(unigram_training_feature.data>1)[0]]
    # new_unigram_training_feature = sparse.csr_matrix(unigram_training_feature)
    # new_unigram_training_feature.data = new_data

    # classifier = NaiveBayesClassifier(training_data, new_training_labels, labels)
    # training_error_rate = classifier.classify(training_data, training_labels)
    # test_error_rate = classifier.classify(test_data, test_labels)
    # print 'training_error_rate: ', training_error_rate
    # print 'test_error_rate: ', test_error_rate
    pass
    


def main():
    start = datetime.now()

    using_num = 200000
    # read training data
    training_data = read_training_data()
    training_corpus = training_data['text'][:using_num]
    training_labels = training_data['label'][:using_num]
    # read test data
    test_data = read_test_data()
    test_corpus = test_data['text']
    test_labels = test_data['label']

    #---------------------- perceptron ----------------------#
    test_perceptron(training_corpus, training_labels, test_corpus, test_labels)
    #---------------------- naive bayes ----------------------#
    # test_naive_bayes()
    
    end = datetime.now()
    used_time = (end - start).seconds
    print 'used time: %d' % used_time


if __name__ == '__main__':
    main()