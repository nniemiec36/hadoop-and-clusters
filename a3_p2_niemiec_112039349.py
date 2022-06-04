# Name: Nicole Niemiec
# SBID: 112039349
# TASK II
from cmath import cos
import sys
from pyspark import SparkContext
import json
import numpy as np

input_file = 'hdfs:/data/review.json'
sc = SparkContext(appName="Homework 3")
sc.setLogLevel("ERROR")

# Step 2.1: 
target_users = sc.broadcast(['PomQayG1WhMxeSl1zohAUA', 'uEvusDwoSymbJJ0auR3muQ', 'q6XnQNNOEgvZaeizUgHTSw', 'n00anwqzOaR52zgMRaZLVQ', 'qOdmye8UQdqloVNE059PkQ'])
# 2.1.1 & 2.1.2:
# 2.1.1: Reach each line into an RDD and get rid of everything except the user_id, business_id, stars (rating), and date. (hint use: .map(json.loads) to get a dictionary per record). Note in this context, "items" are restaurants which are identified by the business_id field in the data.
# 2.1.2: Filter to only one rating per user per item by taking their most recent record. (hint: this is the last time you will need the "date" field)
rdd = sc.textFile(input_file, 32).map(json.loads)
rdd = rdd.map(lambda x: [(x['user_id'], x['business_id']), (x['date'], x['stars'])])
rdd = rdd.reduceByKey(lambda a, b: a if max(a[0], b[0]) == a[0] else b)

# 2.1.3: From there, filter to items (i.e. business_ids) associated with at least 30 distinct users.

businesses = rdd.map(lambda x: [x[0][1], (x[0][0], x[1][1])]).groupByKey().mapValues(list)
businesses_rdd = businesses.filter(lambda x: len(x[1]) >= 30)


# 2.1.4: From there, filter to users associated with at least 5 distinct items.
user_rdd = businesses_rdd.flatMap(lambda x: [(x[1][i][0], (x[0], x[1][i][1])) for i in range(len(x[1]))]).groupByKey().mapValues(list)
user_rdd = user_rdd.map(lambda x: [x[0], {k:v for k,v in x[1]}]).filter(lambda x: len(x[1]) >= 5)


# 2.1.5: Extract the following target users along with their business_id ratings into a broadcast variable named "target_users".
target = user_rdd.filter(lambda x: x[0] in target_users.value)
target = target.collect()
# 2.1 Checkpoint: 
# Print the first 10 business_ids and ratings for the target users. For each user, sort alphanumeric by business_id (this can be done outside the RDD) and only include the first 10 of them.
print("2.1 Checkpoint: \n")

# print(target.take(1))
for user_id, value in target:
   print("USER ID: " + user_id)
   value = sorted(value.items())[:10]
   print(value)
#    print("USER ID: " + user_id)
#    print("[")
#    for i in range(0, 10):
#        if i == 0:
#             print("[")
#        print(value)
#        if i == 9:
#             print("]")


# Step 2.2: Perform user-user collaborative filtering.

# 2.2.1: Read the following step (2.2.2) and transform your RDD into a format(s) appropriate for finding all similar users to the given target_users. (tip: using business_ids as a key makes it easy to find all those who rated a particular business; using another RDD with the user_id as key makes it easy to find all other businesses rated by a given user).
# user_mapping = user_rdd.map(lambda x: (x[0], list(x[1].items()))).flatMap(lambda x: [(x[0], (x[1][i], 1)) for i in range(len(x[1]))])
# business_mapping = user_rdd.map(lambda x: (x[0], list(x[1].items()))).flatMap(lambda x: [(x[1][i][0], (x[0], x[1][i][1])) for i in range(len(x[1]))]).combineByKey(lambda v:[v],lambda x,y:x+[y],lambda x,y:x+y)

# user_maps = user_rdd.map(lambda x: (x[0], list(x[1].items())))
business_maps = user_rdd.map(lambda x: (x[0], list(x[1].items()))).flatMap(lambda x: [(x[1][i][0], (x[0], x[1][i][1])) for i in range(len(x[1]))]).groupByKey().map(lambda x: (x[0], {i[0]:i[1] for i in x[1]}))

# .combineByKey(lambda v:[v],lambda x,y:x+[y],lambda x,y:x+y)

# 2.2.2: For each of the target users, find up to 50 most similar neighbors (i.e. other users).
# Neighbors must:
#        (a) have at least two ratings for the same businesses as the target_users
#        (b) have a positive, non-zero similarity with the target users
# Use the cosine similarity of mean-centered ratings as the similarity metric.

# performs mean centering and maps user_rdd as follows:
# (user_id, {'business_id': mean_centered_rating, 'business_id': mean_centered_rating, ... })
user_maps = user_rdd.map(lambda x: (x[0], x[1], sum(x[1].values())/len(x[1]))).map(lambda x: (x[0], {pair[0]:pair[1]-x[2] for pair in x[1].items()}))

# broadcast variable for target_users formatted like:
# (user_id, {'business_id': mean_centered_rating, 'business_id': mean_centered_rating, ... })
target = sc.broadcast(user_maps.filter(lambda x: x[0] in target_users.value).collect())

temp = user_maps.filter(lambda x: x[0] in target_users.value)
# target but formatted like:
# [[(user_id1, business_id), (user_id1, business_id), ...], [(user_id2, business_id), (user_id2, business_id), ...], ...]
target_pairs = sc.broadcast(temp.map(lambda x: {x[0]: i for i in x[1].keys()}).collect())

# formatted like:
# ('user_id_1', 'user_id_2', {business_id: mean_centered_rating, ...}[x], {business_id: mean_centered_rating, ...}[other])
# modeled from the slides
cos_sim = user_maps.flatMap(lambda other: [(x[0], other[0], x[1], other[1]) for x in target.value])
# filters those with at least two ratings
cos_sim = cos_sim.filter(lambda x: len(set(x[2].keys()).intersection(set(x[3].keys()))) >= 2)

cos_sim = cos_sim.map(lambda x: ((x[0], x[1]), (np.sqrt(np.sum(np.square(list(x[2].values())))), np.sqrt(np.sum(np.square(list(x[3].values()))))), sorted([(k, v) for k, v in x[2].items()] + [(k, v) for k, v in x[3].items()], key=lambda y: y[0])))


cos_sim = cos_sim.map(lambda x: (x[0][0], (x[0][1], sum([x[2][i][1]*x[2][i+1][1] for i in range(len(x[2])-1) if x[2][i][0] == x[2][i+1][0]])/(x[1][0]*x[1][1]))))
cos_sim = cos_sim.filter(lambda x: x[1][1]>0)
cos_sim = cos_sim.filter(lambda x: x[0] != x[1][0])

# takes highest 50 similarities
cos_sim = cos_sim.groupByKey().mapValues(list)
cos_sim = cos_sim.map(lambda x: (x[0], sorted(x[1], key=lambda y: y[1], reverse=True)[:50]))
sim_users = sc.broadcast(cos_sim.collect())
cos_sim = cos_sim.flatMap(lambda x: [(i[0], (x[0], i[1])) for i in x[1]])


user_predictions = business_maps.flatMap(lambda x: [(i[0], (x[0], sum([x[1][u[0]]*u[1] for u in i[1] if u[0] in x[1].keys()])/sum([u[1] for u in i[1] if u[0] in x[1].keys()]))) for i in sim_users.value if i[0] not in x[1].keys() and len(set([j[0] for j in i[1]]).intersection(set(x[1].keys()))) >= 3 ])
user_predictions = user_predictions.groupByKey().mapValues(list)

print("\nCheckpoint 2.2: \n\n")
user_pred = user_predictions.collect()
for user_id, value in user_pred:
    print("USER ID: " + user_id)
    value = sorted(value)[:10]
    print(value)


###################### 2.3 ##########################
# business_rdd = business_maps.map(lambda x: (x[0], {i: (x[1][i] - sum(x[1].values())/len(x[1].values())) for i in x[1].keys()}))

# target_businesses = business_rdd.flatMap(lambda x: [x[0] for i in x[1].keys() if i in target_users.value]).distinct()

# sim_biz = target_businesses.cartesian(target_businesses).filter(lambda x: x[0][0]!=x[1][0] and sum([abs(k) for k in x[0][1].values()!=0 and sum([abs(k) for k in x[1][1].values()])!=0).map(lambda x: (x[0][0], (x[1][0], sum([x[0][1][u]*x[1][1][u] for u in x[]]))))]))

print("\nCheckpoint 2.3: \n\n")




# cos_sim = cos_sim.map(lambda x: (x[0], list(x[1])[:50])).flatMap(lambda x: [(i[0], (x[0], i[1])) for i in x[1]])

# print("User mapping: ")
# print(user_maps.take(1))
# print("Business_mapping: ")
# print(business_maps.take(1))

# 2.2.2: For each of the target users, find up to 50 most similar neighbors (i.e. other users).
# Neighbors must:
#        (a) have at least two ratings for the same businesses as the target_users
#        (b) have a positive, non-zero similarity with the target users
# Use the cosine similarity of mean-centered ratings as the similarity metric.

# def mean_centering(x):
#     # (user_id, [(business_id, rating), (business_id, rating), ...]
#     mean_rating = 0
#     for i in list(x[1]):
#         mean_rating += i[1]
#     mean_rating = mean_rating/len(x[1])
#     mean_centered = [(i[0], i[1] - mean_rating) for i in x[1]]
#     return [(x[0], mean_centered)]

# def find_values(list):
#     n1 = 'WcHGqH9kwTKsvsN_w12cgQ'
#     n2 = 'nRtYC2WjOXFi3HAmAosvNw'
#     for i in list:
#         if i[0][0] == n1 and i[0][1] == n2:
#             return (True, i[1])
#         if i[0][1] == n1 and i[0][0] == n2:
#             return (True, i[1])
#     return (False, 0)




# Step 2.3: Perform item-item collaborative filtering.

# 2.3.1: Transform your RDD (from 2.1) into a format(s) appropriate for finding all similar items to the given all of the items that target_usees reviewed.
# 2.3.2: For each item in the entire dataset, find up to 50 most similar neighbors (i.e. other items).
# Neighbors must:
#        (a) have at least one rating from one of the target users (if not, do not consider the item as a potential neighbor).
#        (b) have a positive, non-zero similarity with the target item.
#        Use the consine similarity of mean-centered ratings as the similarity metric.
# 2.3.3: Make predictions of how the user would rate other resteraunts based on how the user rated similar restaurants. Only make predictions for resteraunts with at least 3 neighbors.

#Checkpoint 2.3

#Print the first 10 business_ids that received a predicted rating along with their predicted rating for the target users. For each user, sort alphanumeric by business_id (this can be done outside the RDD) and only include the first 10.


