import pickle
import math
import sys

from append_features import appendFeatures
sys.path.append("../../tools/")
from feature_format import featureFormat, targetFeatureSplit

"""
    Data Viewer - Use this to export the data to a CSV file. This helps me think about the
    right set of features and machine learning algorithms
"""
DATASET_PICKLE_FILENAME = "../my_dataset.pkl"
OUTPUT_FILENAME = "reports/dataset.csv"

def createDataWorksheet(dataset):

    names = dataset.keys()
    fields = dataset[names[0]].keys()

    print "Names: " , names
    print "Fields: " , fields
    output_file = open(OUTPUT_FILENAME, "w")
    i = 0
    for each_name in names:
        if i == 0:
            headers = "name,"
            headers += ",".join(fields)
            output_file.write(headers + "\n")
        output_line = ""
        output_line += each_name + ","
        for each_field in fields:
            output_value = str(dataset[names[i]][each_field])
            if output_value == 'NaN':
                output_value = "0"
            output_line +=  output_value + ","

        # output_line += ",".join(dataset[names[i]])

        output_line += "\n"
        output_file.write(output_line)
        i += 1
    output_file.close()
    print "File saved as : " + OUTPUT_FILENAME

def load_data():
    with open(DATASET_PICKLE_FILENAME, "r") as dataset_infile:
        dataset = pickle.load(dataset_infile)
    return dataset

def removeStringKeys(dataFields):
    output_list = []
    for each_key in dataFields:
        if each_key != 'email_address':
            output_list.append(each_key)
    return output_list

def main():
    ### load up student's classifier, dataset, and feature_list
    dataset = load_data()
    appendFeatures(dataset)
    createDataWorksheet(dataset)

if __name__ == '__main__':
    main()
