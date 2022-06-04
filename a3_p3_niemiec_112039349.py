# Name: Nicole Niemiec
# SBID: 112039349
# TASK III

import sys
from pyspark import SparkContext, SparkConf
import json
import csv
from collections import Counter
import re
import numpy as np


covid = 'hdfs:/data/test_COVID-19_Hospital_Impact.csv'
dicts = 'hdfs:/data/dictionary.csv'
review = 'hdfs:/data/review.json'
business = 'hdfs:/data/business.json'


# config = SparkConf().setAll([('spark.executor.cores', '4')])
# sc = SparkContext(conf=config)
# sc = SparkContext(appName="Homework 3")
# conf = SparkConf().setAppName(appName).setMaster(master)
sc = SparkContext(appName="Homework 3")
# sc.conf.set("spark.executor.memory", "48g")
# sc.conf.set("spark.executor.cores", "4")
sc.setLogLevel("ERROR")

def create_set(rdd, hospital_bc): 
    rdd = rdd[1:]
    bc = hospital_bc.value[0][1:]
    res = {}
    for value, key in zip(rdd, bc):
        res[key] = value
    return res 


hospitals = []
hospitals_rdd=sc.textFile(covid, 64)
hospitals_rdd=hospitals_rdd.mapPartitions(lambda x: csv.reader(x))
with open("test_COVID-19_Hospital_Impact.csv", "r") as file:
    reader = csv.reader(file)
    for x in reader:
        hospitals.append(x[0])


remove = hospitals_rdd.first()
hospitals_rdd = hospitals_rdd.filter(lambda row: row != remove)

hospital_bc = sc.broadcast(remove)

# (hospital_pk, (zip_code, total_beds, inpatient))
hosps = hospitals_rdd.map(lambda y: (y[0], (y[7], y[11], y[14])))

####################### START 3.1 ##########################
# 3.1: Aggregate outcome data to zip codes.
# 3.1.1:
# Filter to hospital records with at least 30 for total_beds_7_day_avg (for those with smaller total beds, percentages will not be reliable).
hosp = hosps.filter(lambda x: x[1][0] != '' and x[1][0] != "''" and x[1][1] != '' and x[1][1] != "''" and x[1][2] != '' and x[1][2] != "''" and x[1][0] != '-999999' and x[1][0] != '-99999' and x[1][0] != '-9999' and x[1][0] != '-999' and x[1][0] != '-99' and x[1][1] != '-999999' and x[1][1] != '-99999' and x[1][1] != '-9999' and x[1][1] != '-999' and x[1][1] != '-99' and x[1][2] != '-999999' and x[1][2] != '-99999' and x[1][2] != '-9999' and x[1][2] != '-999' and x[1][2] != '-99')
hosp = hosp.map(lambda x: (x[0], (x[1][0], float(x[1][1]), float(x[1][2]))))
hosp = hosp.filter(lambda x: x[1][1] >= 30)
print("End of 3.1.1: \n")

# # 3.1.2:
# Find the mean bed usage percentage per hospital. To do this, first calculate the bed_usage_percent for each record:
hosp = hosp.map(lambda x: (x[0], (x[1][0], x[1][1], x[1][2], x[1][2]/x[1][1])))
hosp_pk = hosp.groupByKey().mapValues(list)
import statistics
def find_mean(list):
    empty_list=[]
    for i in list:
        empty_list.append(i[3])
    return statistics.mean(empty_list)
hosp_pk_means = hosp_pk.map(lambda x: (x[0], find_mean(x[1]), x[1]))

# 3.1.3: 
# Next, aggregate the mean_bed_usage_pct per zip code by taking the mean across all hospitals within that zip code.
# (zip, (hospital_pk, avg))
zips = hosp_pk_means.map(lambda x: (x[2][0][0], (x[1], x[0], x[2])))
zip_codes = zips.groupByKey().mapValues(list)
def find_mean2(list):
    empty_list=[]
    for i in list:
        empty_list.append(i[0])
    return statistics.mean(empty_list)
zip_mean = zip_codes.map(lambda x: (x[0], find_mean2(x[1])))
# print("End of 3.1.3: \n")

####################### CHECKPOINT 3.1 #######################

print("################## CHECKPOINT 3.1 ##################\n")
targets = ['89109', '89118', '15237', '44122', '44106']
target_zips = zip_mean.filter(lambda x: x[0] in targets).map(lambda x: (x[0], x[1])).collect()
# print(target_zips.collect())
for i in target_zips:
    print("ZIP CODE: " + i[0] + " MEAN BED USAGE: " + str(i[1]))
print("######################################################\n\n")

####################### START 3.2 ##########################
# Aggregate the percentage of reviews mentioning each of the 184 dictionary words per zip code. You should not consider how many times a word was mentioned in a particular review but just whether it was mentioned at all.
# usage_score = sum([1 if word mentioned else 0 for all reviews in zip]) / (number of reviews in zip)
# Note: the dictionary allows for prefix matching. Words ending in "*" should be matched to any remaining characters. For example 'alcohol*' matches 'alcohol', 'alcohols', and 'alcoholic' among others.

dict_rdd=sc.textFile(dicts, 64).mapPartitions(lambda x: csv.reader(x))
dictionary = []
with open("dictionary.csv", "r") as file:
    reader = csv.reader(file)
    for x in reader:
        dictionary.append(x[0])

# make this into a broadcast variable
dict_bc = sc.broadcast(dictionary)

biz_rdd=sc.textFile(business, 64).map(json.loads)
rev_rdd=sc.textFile(review, 64).map(json.loads)

biz = biz_rdd.map(lambda x: (x['business_id'], x['postal_code']))
rev = rev_rdd.map(lambda x: (x['business_id'], x['text']))
# 3.2.1:
joined = biz.join(rev)

# 3.2.2:
filtered_reviews = joined.filter(lambda x: len(x[1][1]) >= 256).map(lambda x: (x[1][0], x[1][1])).groupByKey().mapValues(list)

# 3.2.3:
def search_for_words(list):
    count = 0
    list = list.split()
    for reg in dict_bc.value:
        for review in list:
            if re.search(reg, review) != None:
                count += 1
            else:
                count += 0
    return count

regex_bc = sc.broadcast([r"\b((?i)({})(\w+)?)".format(word[:-1]) for word in dictionary if word[-1] == "*"] + [r"\b{}\b".format(word) for word in dictionary if word[-1] != "*"])

reviews = filtered_reviews.map(lambda x: (x[0], len(x[1]), Counter(list(reg.lstrip(r"\b").lstrip(r"((?i)(").rstrip(r"\b").rstrip(r")(\w+)?)") for reg in regex_bc.value for review in x[1] if re.findall(re.compile(reg), review) != []))))
# reviews = filtered_reviews.map(lambda x: (x[0], len(x[1]), search_for_words(x[1])))
# does not work !!!!!!!!!!!
# need to check all the "permuations" of a word
# fry* --> friday, fry, fries, etc. 
reviews = reviews.map(lambda r: (r[0], sorted([(k, v/r[1]) for k,v in dict(r[2]).items()], key=lambda x: x[1], reverse=True)))
reviews_small = reviews.map(lambda x: (x[0], x[1][:5]))

print("################## CHECKPOINT 3.2 ##################\n")
targets = ['89109', '89118', '15237', '44122', '44106']
filtered = reviews_small.filter(lambda x: x[0] in targets).collect()
for i in filtered:
    print("ZIP CODE: " + i[0])
    words = i[1][:5]
    print("WORDS: ")
    print(words)
    print("######################################################\n\n")

##################### START 3.3 ####################
# Restaurants and what we eat differ by and form characterizations of a community.  At the same time the percent of hospital beds used is a strong indicator of burden on a community's health system.  Your objective is to compute the association between 184 ingestion words (case insensitive) and the healthcare burden on a community as measured by mean bed usage percentage of hospitals. You will do this by testing the hypothesis that each of the 184 words is correlated with bed usage, correcting for multiple tests, as well as recording the effect size (Pearson correlation).

# Step 3.3: Calculate correlations between word usage and mean_bed_usage_pct.

# 3.3.1: For each word in the dictionary, correlate its usage with mean_bed_usage_pct.

# word_usage_scores = reviews.map(lambda x: (x[0], {i[0]:i[1] for i in x[1]}))

# usage_by_zipcode = word_usage_scores.join(zip_mean)
# usage_by_word = usage_by_zipcode.flatMap(lambda x: [(word, (x[0], x[1][0][word], x[1][1])) for word in x[1][0].keys()])
# usage_by_word = usage_by_word.groupByKey().map(lambda x: (x[0], [(i[1], i[2]) for i in sorted(x[1], key = lambda k: k[0])])) # sorts by zip code
# normalized_rdd = usage_by_word.map(lambda x: (x[0], ([i[0] for i in x[1]], [i[1] for i in x[1]]))).map(lambda x: (x[0], ([(i-np.mean(x[1][0]))/np.std(x[1][0]) for i in x[1][0]], [(i-np.mean(x[1][1]))/np.std(x[1][1]) for i in x[1][1]])))
# pearson_corrs = normalized_rdd.map(lambda i: (i[0], sum([x*y for x,y in zip(i[1][0], x[1][1])])/(len(x[1][1])-1)))
# print("\n\n3.3.1:")
# print(pearson_corrs.take(2))



# 3.3.2: Calculate the p-value and Bonferonni corrected p-value for each word.
# Checkpoint 3.3

# Print the top 20 most positively correlated words along with their p-value and corrected p-value.

# Print the top 20 most negatively correlated words along with their p-value and corrected p-value.


# joined2 = target_zips.join(filtered)
# words = joined2.map(lambda x: [(i[0], (i[1], x[0], x[1][0])) for i in x[1][1]])
# word_mapping = words.groupByKey().mapValues(list)
