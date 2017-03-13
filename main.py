#!usr/bin/python
# _*_ coding: utf-8 _*_

import bz2
import time

import numpy as np
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_extraction import FeatureHasher
from sklearn.linear_model import SGDClassifier


def chunk_file(f, chunksize=1024000):
    return iter(lambda: f.readlines(chunksize), [])


def feature_hash(X, n_features=1000):
    h = FeatureHasher(n_features=n_features)
    return h.transform(X)


def parse_lines(lines):
    y = []
    X = []
    for line in lines:
        line = line.decode('utf-8').split()
        if len(line) == 1:
            X.append({feature: line[0].split(',').count(feature) for feature in set(line[0].split(','))})
        else:
            y.append(int(line[0]))
            X.append({feature: line[1].split(',').count(feature) for feature in set(line[1].split(','))})
    return (y, X)


def classifier_comparision(train_data_path='training-data-small.txt.bz2', cv=10):
    """This function simply does a comparision of selected classifiers as follows,
    1. support vector machine, kernel: linear
    2. supoort vector machine, kernel: rbf
    3. Nearest Neighbors
    4. Neural Network, MLPClassfier

    :type train_data_path: str
    :params train_data_path: the path to training data file
    :type cv: int
    :params cv: cross-validation generator, ref. k-fold

    :type return: list of tuples
    :params return: a list of tuples, (cross valdiation score, algorithm name)

    :the print-out result of classifier comparision
    - Accuracy of linear svm: 0.67 (+/- 0.0)
    - Accuracy of rbf svm: 0.64 (+/- 0.0)
    - Accuracy of nearest neighbors: 0.56 (+/- 0.0)
    - Accuracy of neural network: 0.67 (+/- 0.0)
    """
    from sklearn import metrics
    from sklearn.model_selection import cross_val_score

    # load training data
    train_X, train_y = load_data(data_path=train_data_path)

    # select classifiers
    classifiers = [(SVC(kernel='linear'), 'linear svm'),
                   (SVC(kernel='rbf'), 'rbf svm'),
                   (KNeighborsClassifier(n_neighbors=3), 'nearest neighbors'),
                   (MLPClassifier(alpha=1), 'neural network')]

    scores = [(cross_val_score(clf, train_X, train_y, cv=cv, scoring='f1'), name) for clf, name in classifiers]
    for score, name in scores:
        print("Accuracy of {}: {} (+/- {})".format(name, round(score.mean(), 2), round(score.std()*2), 2))

    return scores


def load_data(data_path):
    """Read bz2 file and ouput the y and X"""
    with bz2.open(data_path, 'r') as f:
        train_y, X = parse_lines(f.readlines())
        train_X = feature_hash(X)
    return (train_X, train_y)


def optimize_svm(train_data_path='training-data-small.txt.bz2'):
    """Run grid search to determine best C or gamma for svm
    Generally, C ranges from 1 to 1000, and gamma is no larger than 0.1.
    :type train_data_path: str
    :params train_data_path: the path to training data file

    :type return: dict
    :params return: best params obtained by grid search
    """
    from sklearn import grid_search
    from sklearn import metrics

    # load training data
    train_X, train_y = load_data(data_path=train_data_path)

    # config the range of C and gamma in grid search
    param_grid = [{'C': [2**i for i in range(0, 10, 1)],  # 1 <= C <= 1000
                   'gamma': [2**i for i in np.arange(-8, -3, 0.5)],  # 0 < gamma <= 0.1
                   'kernel': ['rbf']},
                  {'C': [2**i for i in range(0, 10, 1)],  # 1 <= C <= 1000
                   'kernel': ['linear']}]
    method = SVC()
    grid_search = grid_search.GridSearchCV(method, param_grid, scoring='f1', n_jobs=9)
    grid_search.fit(train_X, train_y)

    return grid_search.best_params_


def build_model(train_data_path='training-data-small.txt.bz2',
                scale='small',
                C=1,
                gamma=0.1,
                kernel='rbf',
                chunksize=1024000):
    """Return the trained model
    by given training data and parameters (optimized C, gamma)
    """
    if scale == 'small':
        if 'large' in train_data_path:
            raise ValueError("You can only choose small dataset in small scale")
        else:
            model = SVC(C=C, gamma=gamma, kernel=kernel)
            # load training data
            train_X, train_y = load_data(data_path=train_data_path)
            model.fit(train_X, train_y)
    else:
        if 'small' in train_data_path:
            raise ValueError("You can only choose large dataset in large scale")
        else:
            model = SGDClassifier()
            from sklearn.kernel_approximation import RBFSampler
            # kernel approximation
            rbf_feature = RBFSampler(gamma=gamma, random_state=1, n_components=1000)
            # incremental learning with SGDClassifier
            with bz2.open(train_data_path, 'r') as f:
                for chunk in chunk_file(f):
                    print(time.time())
                    train_y, X = parse_lines(chunk)
                    train_X = feature_hash(X)
                    train_X_rbf = rbf_feature.fit_transform(train_X)
                    model.partial_fit(train_X_rbf, train_y, classes=np.array([0, 1]))

    return model


def predict(scale='small', output_file='output-small.txt',
            C=None, gamma=None, kernel=None):
    """Run the prediction with given test data
    """
    # do params optimization
    print("Notice that, the grid search may take long time. In such condition, you may want to use the given C, gamma and kernel")  # nopep8
    if C and gamma and kernel:
        print('hello! Lets go')
        best_C = C
        best_gamma = gamma
        best_kernel = kernel
    else:
        params = optimize_svm()
        best_C = params['C']
        best_gamma = params['gamma']
        best_kernel = params['kenel']

    if scale == 'small':
        # build the model
        model = build_model(C=best_C,
                            gamma=best_gamma,
                            kernel=best_kernel)
        # load test data
        test_X, _ = load_data(data_path='test-data-small.txt.bz2')

    elif scale == 'large':
        model = build_model(train_data_path='training-data-large.txt.bz2',
                            scale='large',
                            gamma=best_gamma,
                            chunksize=1024000)
        # load test data
        test_X, _ = load_data(data_path='test-data-large.txt.bz2')

    # predict result
    predict_y = model.predict(test_X)
    # save result to a file
    with open(output_file, 'w+') as f:
        for y in predict_y:
            print(y, file=f)


if __name__ == '__main__':
    predict(scale='large', output_file='output-large.txt', C=16, gamma=0.0781, kernel='rbf')

