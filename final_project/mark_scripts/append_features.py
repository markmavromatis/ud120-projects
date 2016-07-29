import math

'''
    Add scaled attributes for financial and email statistics
'''
def getBonusSalaryRatio(bonusAmount, salaryAmount):
    if str(bonusAmount) != 'NaN' and str(salaryAmount) != 'NaN' and salaryAmount!= 0:
        return 1.0 * bonusAmount / salaryAmount
    else:
        return 0

def getBonusTotalCompensationRatio(bonusAmount, totalCompAmount):
    if str(bonusAmount) != 'NaN' and str(totalCompAmount) != 'NaN' and totalCompAmount!= 0:
        return 1.0 * bonusAmount / totalCompAmount
    else:
        return 0


def scaleEmails(subsetCount, totalCount):
    if str(subsetCount) != 'NaN' and totalCount != 0:
        return 1.0 * subsetCount / totalCount
    else:
        return 0

def scaleExercisedStock(exercisedStock, totalStock):
    if str(exercisedStock) != 'NaN' and str(totalStock) != 'NaN' and totalStock != 0:
        return 1.0 * exercisedStock / totalStock
    else:
        return 0


def appendFeatures(dataset):
    # Iterate over every row
    for each_key in dataset:
        # Scale mails from POI based on total inbound emails
        datarow = dataset[each_key]
        datarow["from_poi_ratio"] = scaleEmails(datarow["from_poi_to_this_person"], datarow["to_messages"])
        datarow["to_poi_ratio"] = scaleEmails(datarow["from_this_person_to_poi"], datarow["from_messages"])
        datarow["shared_receipt_poi_ratio"] = scaleEmails(datarow["shared_receipt_with_poi"], datarow["to_messages"])
        # How much of total stock holdings was exercised?
        datarow["exercised_stock_percent"] = scaleExercisedStock(datarow["exercised_stock_options"], datarow["total_stock_value"])
        # What is the ratio of bonus to salary? High bonus may indicate strange incentives?
        datarow["bonus_salary_ratio"] = getBonusSalaryRatio(datarow["bonus"], datarow["salary"])

        # What is the ratio of bonus to total compensation?
        datarow["bonus_total_ratio"] = getBonusTotalCompensationRatio(datarow["bonus"], datarow["total_payments"])

