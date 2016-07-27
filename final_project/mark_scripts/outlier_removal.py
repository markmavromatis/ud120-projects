'''
This file is responsible for removing outliers from the Enron dataset
'''


# Remove outliers due to bad data including
def removeOutliers(dataset):
    # Totals row is obviously just the totals from all columns. We need to remove this invalid data.
    del dataset['TOTAL']
    # Some travel agency. This does not reflect a PERSON so we should remove it from the dataset
    del dataset['THE TRAVEL AGENCY IN THE PARK']

