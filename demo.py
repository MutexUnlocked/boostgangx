#    Copyright 2021 Boostgang.

from __future__ import print_function, division

import sys

import numpy as np

from sklearn import preprocessing, metrics
from sklearn.ensemble import AdaBoostClassifier
from sklearn.datasets import load_breast_cancer, fetch_openml
from sklearn.impute import SimpleImputer
from dwave.system.samplers import DWaveSampler
from sklearn.model_selection import train_test_split
from dwave.system.composites import EmbeddingComposite

from qboost import WeakClassifiers, QBoostClassifier


def metric(y, y_pred):

    return metrics.accuracy_score(y, y_pred)


def print_accuracy(y_train, y_train_pred, y_test, y_test_pred):
    """Print information about accuracy."""

    print('    Accuracy on training set: {:5.2f}'.format(metric(y_train, y_train_pred)))
    print('    Accuracy on test set:     {:5.2f}'.format(metric(y_test, y_test_pred)))


def train_models(X_train, y_train, X_test, y_test, lmd, verbose=False):
    NUM_READS = 3000
    NUM_WEAK_CLASSIFIERS = 35
    # lmd = 0.5
    TREE_DEPTH = 3

    # define sampler
    dwave_sampler = DWaveSampler()
    emb_sampler = EmbeddingComposite(dwave_sampler)

    N_train = len(X_train)
    N_test = len(X_test)

    print('Size of training set:', N_train)
    print('Size of test set:    ', N_test)
    print('Number of weak classifiers:', NUM_WEAK_CLASSIFIERS)
    print('Tree depth:', TREE_DEPTH)


    # input: dataset X and labels y (in {+1, -1}

    # Preprocessing data
    scaler = preprocessing.StandardScaler()     # standardize features
    normalizer = preprocessing.Normalizer()     # normalize samples

    X_train = scaler.fit_transform(X_train)
    X_train = normalizer.fit_transform(X_train)

    X_test = scaler.fit_transform(X_test)
    X_test = normalizer.fit_transform(X_test)


    # ===============================================
    print('\nAdaboost:')

    clf = AdaBoostClassifier(n_estimators=NUM_WEAK_CLASSIFIERS)

    clf.fit(X_train, y_train)

    hypotheses_ada = clf.estimators_
    y_train_pred = clf.predict(X_train)
    y_test_pred = clf.predict(X_test)

    print_accuracy(y_train, y_train_pred, y_test, y_test_pred)


    # ===============================================
    print('\nDecision tree:')

    clf2 = WeakClassifiers(n_estimators=NUM_WEAK_CLASSIFIERS, max_depth=TREE_DEPTH)
    clf2.fit(X_train, y_train)

    y_train_pred2 = clf2.predict(X_train)
    y_test_pred2 = clf2.predict(X_test)

    if verbose:
        print('weights:\n', clf2.estimator_weights)

    print_accuracy(y_train, y_train_pred2, y_test, y_test_pred2)


    # ===============================================
    print('\nQBoost:')

    DW_PARAMS = {'num_reads': NUM_READS,
                 'auto_scale': True,
                 'num_spin_reversal_transforms': 10,
                 }

    clf3 = QBoostClassifier(n_estimators=NUM_WEAK_CLASSIFIERS, max_depth=TREE_DEPTH)
    clf3.fit(X_train, y_train, emb_sampler, lmd=lmd, **DW_PARAMS)

    y_train_dw = clf3.predict(X_train)
    y_test_dw = clf3.predict(X_test)

    if verbose:
        print('weights\n', clf3.estimator_weights)

    print_accuracy(y_train, y_train_dw, y_test, y_test_dw)


if __name__ == '__main__':

    print('Solar Flae:')

    X,y = fetch_openml('solar-flare', version=1, return_X_y=True)

    # train on a random 2/3 and test on the remaining 1/3

    X_train, X_test, y_train, y_test = train_test_split (X,y, test_size = 0.3, random_state = 0)

    print(y_train)
    y_train = 2*(y_train == '1') - 1
    y_test = 2*(y_test == '1') - 1

    print(y_train)
    train_models(X_train, y_train, X_test, y_test, 1.0)

