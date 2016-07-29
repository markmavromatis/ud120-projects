#!/usr/bin/pickle

"""
Modified version of tester that iterates through features and tries different combinations of them
against different classifiers.

This script tests all iterations of 2-feature classifiers
"""

import pickle
import sys
import numpy
from append_features import appendFeatures
from outlier_removal import removeOutliers

from sklearn.cross_validation import StratifiedShuffleSplit
# Naive Bayes Classifier
from sklearn.naive_bayes import GaussianNB
# Support Vector Machine
from sklearn.svm import SVC
# Decision Tree Classifier
from sklearn import tree
# Random Forest Estimator
from sklearn.ensemble import RandomForestClassifier

sys.path.append("../../tools/")
from feature_format import featureFormat, targetFeatureSplit

OUTPUT_FILENAME = "reports/2feature_results.csv"

PERF_FORMAT_STRING = "\
\tAccuracy: {:>0.{display_precision}f}\tPrecision: {:>0.{display_precision}f}\t\
Recall: {:>0.{display_precision}f}\tF1: {:>0.{display_precision}f}\tF2: {:>0.{display_precision}f}"
RESULTS_FORMAT_STRING = "\tTotal predictions: {:4d}\tTrue positives: {:4d}\tFalse positives: {:4d}\
\tFalse negatives: {:4d}\tTrue negatives: {:4d}"

PERF_FORMAT_CSV = "\
, {:>0.{display_precision}f}, {:>0.{display_precision}f}, {:>0.{display_precision}f}, {:>0.{display_precision}f}, {:>0.{display_precision}f}"
RESULTS_FORMAT_CSV = ", {:4d}, {:4d}, {:4d}, {:4d}, {:4d}"


def test_classifier(clf, dataset, feature_list, results_summary, folds=1000):
    data = featureFormat(dataset, feature_list, sort_keys=True)
    labels, features = targetFeatureSplit(data)
    cv = StratifiedShuffleSplit(labels, folds, random_state=42)
    true_negatives = 0
    false_negatives = 0
    true_positives = 0
    false_positives = 0
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

        ### fit the classifier using training set, and test on test set
        clf.fit(features_train, labels_train)
        predictions = clf.predict(features_test)
        for prediction, truth in zip(predictions, labels_test):
            if prediction == 0 and truth == 0:
                true_negatives += 1
            elif prediction == 0 and truth == 1:
                false_negatives += 1
            elif prediction == 1 and truth == 0:
                false_positives += 1
            elif prediction == 1 and truth == 1:
                true_positives += 1
            else:
                print "Warning: Found a predicted label not == 0 or 1."
                print "All predictions should take value 0 or 1."
                print "Evaluating performance for processed predictions:"
                break
    try:
        total_predictions = true_negatives + false_negatives + false_positives + true_positives
        accuracy = 1.0 * (true_positives + true_negatives) / total_predictions
        precision = 1.0 * true_positives / (true_positives + false_positives)
        recall = 1.0 * true_positives / (true_positives + false_negatives)
        f1 = 2.0 * true_positives / (2 * true_positives + false_positives + false_negatives)
        f2 = (1 + 2.0 * 2.0) * precision * recall / (4 * precision + recall)
        classifierType = ""
        if isinstance(clf, GaussianNB):
            classifierType = "Naive Bayes"
        elif isinstance(clf, tree.DecisionTreeClassifier):
            classifierType = "Decision Tree"
        elif isinstance(clf, RandomForestClassifier):
            classifierType = "Random Forest"
        elif isinstance(clf, SVC):
            classifierType = "Support Vector Machine"
        else:
            raise Exception("Unsupported classifier type: " + type(clf))
        results_summary.append(classifierType + ", " + ', '.join(feature_list[1:]) + ", "
                               + PERF_FORMAT_CSV.format(accuracy, precision, recall, f1, f2, display_precision=5))

        print clf
        print PERF_FORMAT_STRING.format(accuracy, precision, recall, f1, f2, display_precision=5)
        print RESULTS_FORMAT_STRING.format(total_predictions, true_positives, false_positives, false_negatives,
                                           true_negatives)
        print ""
    except:
        print sys.exc_info()[0]
        # raise
        print "Got a divide by zero when trying out:", clf
        print "Precision or recall may be undefined due to a lack of true positive predicitons."


DATASET_PICKLE_FILENAME = "../my_dataset.pkl"

def load_data():
    with open(DATASET_PICKLE_FILENAME, "r") as dataset_infile:
        dataset = pickle.load(dataset_infile)
    return dataset


def main():
    ### load up student's classifier, dataset, and feature_list
    dataset = load_data()
    classifiers = ["NaiveBayes", "SupportVectorMachine", "DecisionTree", "RandomForest"]

    # Final iteration
    important_features = ['total_payments','total_stock_value','exercised_stock_options','bonus','restricted_stock','shared_receipt_with_poi','to_messages',
                          'from_messages','from_poi_ratio','shared_receipt_poi_ratio','to_poi_ratio','bonus_salary_ratio','exercised_stock_percent','bonus_total_ratio']

    print "# Rows = " , len(dataset)


    results_summary = []
    results_summary.append("Classifier,Field1,Field2,,accuracy, precision, recall, f1, f2")
    # Pick 2 of each item in the important features list
    for each_classifier in classifiers:
        clf = 0
        if each_classifier == "NaiveBayes":
            clf = GaussianNB()
        elif each_classifier == "DecisionTree":
            clf = tree.DecisionTreeClassifier()
        elif each_classifier == "RandomForest":
            clf = RandomForestClassifier(n_estimators=10)
        elif each_classifier == "SupportVectorMachine":
            clf = SVC()
        else:
            raise Exception("Unsupported classifier type: " + each_classifier)


        # 2-feature test
        for i in range(len(important_features) - 1):
            for j in range(i + 1, len(important_features)):

                iterate_features = [important_features[i], important_features[j]]
                # print "Testing NB classifier for features: " + str(iterate_features)
                classifier_features = ["poi"] + iterate_features
                print "CLF Features: " + str(classifier_features)
                test_classifier(clf, dataset, classifier_features, results_summary)

    outfile = open(OUTPUT_FILENAME, "w")
    print "Here are the results:"
    outfile.write("\n".join(results_summary))
    outfile.close()


    print("Saved results to file: " + OUTPUT_FILENAME)

if __name__ == '__main__':
    main()
