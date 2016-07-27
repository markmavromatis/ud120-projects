import pickle
import math

"""
    Data Viewer - Use this to look at the data while thinking about the
    right set of features and machine learning algorithms
"""
DATASET_PICKLE_FILENAME = "../my_dataset.pkl"
OUTPUT_FILENAME = "reports/dataset.csv"

def createDataWorksheet(dataset):

    output_file = open(OUTPUT_FILENAME, "w")
    i = 0
    for each_key in dataset:
        if i == 0:
            headers = ["name"]
            headers += dataset[each_key].keys()
            output_file.write(",".join(headers) + "\n")
        output_line = ""
        output_line += each_key + ","
        for each_element in dataset[each_key]:
            output_element = dataset[each_key][each_element]
            if str(output_element) == 'NaN':
                output_element = ""
            output_line += str(output_element) + ","
        output_line += "\n"            
        output_file.write(output_line)
        i += 1
    output_file.close()
    print "File saved as : " + OUTPUT_FILENAME

def load_data():
    with open(DATASET_PICKLE_FILENAME, "r") as dataset_infile:
        dataset = pickle.load(dataset_infile)
    return dataset


def main():
    ### load up student's classifier, dataset, and feature_list
    dataset = load_data()
    createDataWorksheet(dataset)

if __name__ == '__main__':
    main()
