import pickle
import operator
from outlier_removal import removeOutliers
DATASET_PICKLE_FILENAME = "../my_dataset.pkl"
THRESHOLD = 0.5

def load_data():
    with open(DATASET_PICKLE_FILENAME, "r") as dataset_infile:
        dataset = pickle.load(dataset_infile)
    return dataset

def getStats(dataset):
    row_count = 0
    stats = {}

    for each_person in dataset:
        row_count += 1
        each_row = dataset[each_person]
        for each_key in each_row:
            if not each_key in stats:
                stats[each_key] = 0
            if str(each_row[each_key]) != 'NaN':
                stats[each_key] += 1

    # Remove email address and POI (non-numeric values)
    del stats['poi']
    del stats['email_address']

    # Copied code from super helpful StackOverflow post:
    # http://stackoverflow.com/questions/613183/sort-a-python-dictionary-by-value
    sorted_stats = sorted(stats.items(), key=operator.itemgetter(1), reverse=True)
    top_stats = []
    for each_row in sorted_stats:
        percentage_of_total = 1.0 * each_row[1] / row_count
        # print each_row[0] + ": " + str(percentage_of_total) + "%"
        if percentage_of_total >= THRESHOLD:
            top_stats.append([each_row[0], percentage_of_total])
    return top_stats

def displayResults(top_fields):
    print "Fields with over " + str(THRESHOLD * 100) + "% available data:"
    for each_row in top_fields:
        formatted_percentage = round(each_row[1] * 100, 2)
        print each_row[0] + ": " + str(formatted_percentage) + "%"


def main():
    # load up student's classifier, dataset, and feature_list
    dataset = load_data()
    # Remove outlier records
    removeOutliers(dataset)
    # Extract fields populated for at least 50% of records
    top_fields = getStats(dataset)
    displayResults(top_fields)

if __name__ == '__main__':
    main()
