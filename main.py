import bz2
import numpy as np
from sklearn import svm
from sklearn import grid_search
from sklearn import cross_validation
from sklearn.feature_extraction import DictVectorizer, FeatureHasher
import argparse
import logging
import datetime as dt
import csv

def init_logger():
    global logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    log_fmt = '%(asctime)s/%(name)s[%(levelname)s]: %(message)s'
    logging.basicConfig(format=log_fmt)

def read_bz2(file_path):
    """read file directly from bz2 file, return tuple of list of y and list of X
    :type file_path: str
    :type all_X/x: list
    """
    all_X = []
    all_y = []
    with bz2.open(file_path, 'r') as f:
        for line in f:
            y, X = parse_line(line.decode('utf-8'))
            all_X.append(X)
            all_y.append(y)
    return (all_y, all_X)


def parse_line(line):
    r"""parse a single line, return tuple (y, X)
    :type line: str

    :type y: int, 0 or 1
    :type X: dict, number of occurances of each feature
    >>> parse_line('0\ta,a,a\n')
    (0, {'a': 3})
    """
    # get y
    if (len(line.split())==2):
        y = int(line.split()[0])
        X_list = line.split()[1].split(',')
    else:
        y = None
        X_list = line.split()[0].split(',')
    # count occurances of each unique feature in line
    X = {feature: X_list.count(feature) for feature in set(X_list)}
    return (y, X)


def hasher(training_file, test_file):
    """
    :function: transform str to vector for features
    choose the NO of feature for hash table to keep the load factor at 75% or less
    (test and see how many collisions by changing the NO and hash function)
    X_train, Y_test are feature lists
    :type train, test: matrix
    """
    h = FeatureHasher(n_features=1000)
    X_train = read_bz2(training_file)[1]
    X_test = read_bz2(test_file)[1]
    train_vector = h.transform(X_train)
    test_vector = h.transform(X_test)
    return (train_vector, test_vector)


def get_optimized_estimator_by_grid_search(train_X, train_y):

    params = [
        {'C': [2**i for i in range(2, 3, 1)], 'gamma': [2**i for i in np.arange(-8, -7.5, 0.5)], 'kernel': ['rbf']},
    ]
    method = svm.SVC()
    gscv = grid_search.GridSearchCV(method, params, scoring='accuracy', n_jobs=9)
    gscv.fit(train_X, train_y)
    for params, mean_score, all_scores in gscv.grid_scores_:
        logger.info('{:.3f} (+/- {:.3f}) for {}'.format(mean_score, all_scores.std() / 2, params))
    logger.info('params:{params}'.format(params=gscv.best_params_))
    logger.info('score:{params}'.format(params=gscv.best_score_))


if __name__ == '__main__':
    start_time = dt.datetime.now()
    init_logger()
    y, X = read_bz2('training-data-large.txt.bz2')
    train_v, test_v = hasher(training_file='training-data-large.txt.bz2',
                             test_file='test-data-large.txt.bz2')
    get_optimized_estimator_by_grid_search(train_X=train_v, train_y=y)
    end_time = dt.datetime.now()
    print('Total time: {}'.format((end_time - start_time).seconds))
