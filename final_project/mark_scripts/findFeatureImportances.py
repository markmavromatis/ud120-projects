#!/usr/bin/pickle

""" a basic script for importing student's POI identifier,
    and checking the results that they get from it

    requires that the algorithm, dataset, and features list
    be written to my_classifier.pkl, my_dataset.pkl, and
    my_feature_list.pkl, respectively

    that process should happen at the end of poi_id.py
"""

import pickle
import sys
from sklearn.cross_validation import StratifiedShuffleSplit
sys.path.append("../../tools/")
from feature_format import featureFormat, targetFeatureSplit

CLF_PICKLE_FILENAME = "../my_classifier.pkl"
DATASET_PICKLE_FILENAME = "../my_dataset.pkl"
FEATURE_LIST_FILENAME = "../my_feature_list.pkl"


def load_classifier_and_data():
    with open(CLF_PICKLE_FILENAME, "r") as clf_infile:
        clf = pickle.load(clf_infile)
    with open(DATASET_PICKLE_FILENAME, "r") as dataset_infile:
        dataset = pickle.load(dataset_infile)
    with open(FEATURE_LIST_FILENAME, "r") as featurelist_infile:
        feature_list = pickle.load(featurelist_infile)
    return clf, dataset, feature_list

def main():
    ### load up student's classifier, dataset, and feature_list
    clf, dataset, feature_list = load_classifier_and_data()
    data = featureFormat(dataset, feature_list, sort_keys = True)
    labels, features = targetFeatureSplit(data)
    cv = StratifiedShuffleSplit(labels, 1000, random_state=42)

    # Build an empty feature importance totals array for calculating average importance
    totals = []
    for each_feature in feature_list:
        totals.append(0)

    for train_idx, test_idx in cv:
        features_train = []
        features_test = []
        labels_train = []
        labels_test = []
        for ii in train_idx:
            features_train.append(features[ii])
            labels_train.append(labels[ii])
        for jj in test_idx:
            features_test.append(features[jj])
            labels_test.append(labels[jj])
        clf = clf.fit(features_train, labels_train)
        for i in range(len(clf.feature_importances_)):
            totals[i] += clf.feature_importances_[i]
        # print clf.feature_importances_

    for i in range(len(totals)):
        totals[i] /= 1000

    # Display results
    print "Feature list: ", feature_list[1:]
    print "Importances: ", totals

                # # print"Labels = ", labels
    # # print"Features = ", features
    # print "Running decision tree classifier on all data"
    # print"Feature List = ", feature_list[1:]
    # clf = clf.fit(features, labels)

if __name__ == '__main__':
    main()
