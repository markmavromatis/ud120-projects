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
from sklearn import grid_search
from sklearn.tree import DecisionTreeClassifier
from append_features import appendFeatures
from outlier_removal import removeOutliers

PERF_FORMAT_STRING = "\
\tAccuracy: {:>0.{display_precision}f}\tPrecision: {:>0.{display_precision}f}\t\
Recall: {:>0.{display_precision}f}\tF1: {:>0.{display_precision}f}\tF2: {:>0.{display_precision}f}"
RESULTS_FORMAT_STRING = "\tTotal predictions: {:4d}\tTrue positives: {:4d}\tFalse positives: {:4d}\
\tFalse negatives: {:4d}\tTrue negatives: {:4d}"

def test_classifier(clf, dataset, feature_list, summaries, folds = 1000):
    data = featureFormat(dataset, feature_list, sort_keys = True)
    labels, features = targetFeatureSplit(data)
    cv = StratifiedShuffleSplit(labels, folds, random_state = 42)
    true_negatives = 0
    false_negatives = 0
    true_positives = 0
    false_positives = 0
    for train_idx, test_idx in cv:
        features_train = []
        features_test  = []
        labels_train   = []
        labels_test    = []
        for ii in train_idx:
            features_train.append( features[ii] )
            labels_train.append( labels[ii] )
        for jj in test_idx:
            features_test.append( features[jj] )
            labels_test.append( labels[jj] )

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
        accuracy = 1.0*(true_positives + true_negatives)/total_predictions
        precision = 1.0*true_positives/(true_positives+false_positives)
        recall = 1.0*true_positives/(true_positives+false_negatives)
        f1 = 2.0 * true_positives/(2*true_positives + false_positives+false_negatives)
        f2 = (1+2.0*2.0) * precision*recall/(4*precision + recall)
        summaries[len(summaries) - 1] += "" + str(accuracy) + "," + str(precision) + "," +\
                str(recall) + "," + str(f1) + "," + str(f2)
        print "Accuracy = ", accuracy, "Precision = ", precision, "Recall =", recall
        print ""
    except:
        print sys.exc_info()[0]
        print ""
        print "Got a divide by zero when trying out:", clf
        print "Precision or recall may be undefined due to a lack of true positive predicitons."

DATASET_PICKLE_FILENAME = "../my_dataset.pkl"
FEATURE_LIST_FILENAME = "../my_feature_list.pkl"
OUTPUT_FILENAME = "reports/DTParametersReport.csv"

def load_features_and_data():
    with open(DATASET_PICKLE_FILENAME, "r") as dataset_infile:
        dataset = pickle.load(dataset_infile)
    with open(FEATURE_LIST_FILENAME, "r") as featurelist_infile:
        feature_list = pickle.load(featurelist_infile)
    return dataset, feature_list

def main():
    ### load up student's classifier, dataset, and feature_list
    dataset, feature_list = load_features_and_data()

    # Iteration classifier tuning parameters
    max_feature_options = (1, 2, 3)
    max_depth_options = (None, 5, 6, 7, 8, 9, 10)
    min_samples_split_options = (2, 4, 6, 8)

    results_summary = []
    results_summary.append("max_features, max_depth, min_samples_split,, accuracy, precision, recall, f1, f2")

    # Run classifier tests for each parameter setting.
    for each_max_feature_option in max_feature_options:
        for each_depth_option in max_depth_options:
            for each_min_samples_option in min_samples_split_options:
                clf = DecisionTreeClassifier(max_features = each_max_feature_option, max_depth = each_depth_option, min_samples_split = each_min_samples_option)
                resultsSummaryHeader = "" + str(each_max_feature_option) + "," + str(each_depth_option) + "," + str(each_min_samples_option) + ",,"
                results_summary.append(resultsSummaryHeader)
                test_classifier(clf, dataset, feature_list, results_summary)

    print("\n".join(results_summary))
    print("Done")

    outfile = open(OUTPUT_FILENAME, "w")
    outfile.write("\n".join(results_summary))
    outfile.close()

    print("Saved results to file: " + OUTPUT_FILENAME)
    # print "Best parameters = ",clf.best_params_

if __name__ == '__main__':
    main()
