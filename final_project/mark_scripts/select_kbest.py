import pickle
import sys
import numpy as np
import operator

from append_features import appendFeatures
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from missing_data_stats import getStats
from outlier_removal import removeOutliers

sys.path.append("../../tools/")
from feature_format import featureFormat, targetFeatureSplit

"""
    Data Viewer - Use this to look at the data while thinking about the
    right set of features and machine learning algorithms
"""
DATASET_PICKLE_FILENAME = "../my_dataset.pkl"
OUTPUT_FILENAME = "reports/dataset.csv"

def runSelectKBest(k_value, dataset, feature_list):
    # print("Dataset = " ,dataset)
    # print("Feature List = " , feature_list)
    data = featureFormat(dataset, feature_list, sort_keys=True)
    labels, features = targetFeatureSplit(data)
    # print("Labels = ", labels)
    # print("Features = " , features)
    new_features = SelectKBest(chi2, k=k_value)
    new_features.fit_transform(features, labels)

    print "**********"
    featureScores = {}
    for i in range(len(feature_list) - 1):
        # print "I = ", i
        # print "Feature label: ", feature_list[i+1]
        # print "Feature score: ", new_features.scores_[i]
        featureScores[feature_list[i + 1]] = new_features.scores_[i]

    sorted_scores = sorted(featureScores.items(), key=operator.itemgetter(1), reverse=True)
    # print sorted_scores
    for each_row in sorted_scores:
        print each_row
    #     print sorted_scores

def load_data():
    with open(DATASET_PICKLE_FILENAME, "r") as dataset_infile:
        dataset = pickle.load(dataset_infile)
    return dataset


def main():
    ### load up student's classifier, dataset, and feature_list
    dataset = load_data()
    # Remove outlier records
    removeOutliers(dataset)

    # SelectKBest only works with positive values.
    # Apparently some negative numbers are causing problems....

    # Why does this guy have no salary and negative restricted stock?
    # He seems like an outleir
    del dataset['BHATNAGAR SANJAY']
    # He has negative stock too
    del dataset['BELFER ROBERT']

    appendFeatures(dataset)
    top_stats = getStats(dataset)
    print("Top stats = ", top_stats)
    runSelectKBest(5, dataset, ['poi'] + top_stats)

if __name__ == '__main__':
    main()
