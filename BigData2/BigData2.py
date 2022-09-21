# Databricks notebook source
## Big Data Management & Analytics
## Part 1 - PageRank for Airports
# Mounting S3 Bucket
access_key = ""
secret_key = ""
encoded_secret_key = secret_key.replace("/", "%2F")
aws_bucket_name = "stanley-assignment2"
mount_name = "S3_Assignment2"

#dbutils.fs.mount("s3a://%s:%s@%s" % (access_key, encoded_secret_key, aws_bucket_name), "/mnt/%s" % mount_name)
display(dbutils.fs.ls("/mnt/%s" % mount_name)) # Uncomment above line to remount

# COMMAND ----------

pageRankInputParams = sc.textFile("/mnt/S3_Assignment2/AirportPageRankInput.txt")
airportSourceDataLocation = pageRankInputParams.take(3)[0] # S3 location of input file 
pageRankIterations = int(pageRankInputParams.take(3)[1]) #Input file has iteration count = 10
airportRankSaveDestination = pageRankInputParams.take(3)[2] # S3 location to save output file

# COMMAND ----------

S3DF = spark.read.csv(airportSourceDataLocation, header=True, inferSchema= True)
display(S3DF)

# COMMAND ----------

# Minimizing dataframe to only contain nodes (airports) and edges (outlinks)
minimizedAirportDF = S3DF.select(["ORIGIN_AIRPORT_ID", "DEST_AIRPORT_ID"]) 
inputAirportRDD = minimizedAirportDF.rdd
inputAirportRDD.collect()

# COMMAND ----------

def to_list(a):
    return [a]

def append(a, b):
    a.append(b)
    return a

def extend(a, b):
    a.extend(b)
    return a

# Creating pair RDDs containing origin airport as key and list of all outlinks as value
groupedAirportRDD = inputAirportRDD.combineByKey(to_list, append, extend)
groupedAirportRDD.collect()

# COMMAND ----------

# Getting count of distinct airports for later use in PageRank computation
distinctAirportCount = inputAirportRDD.keys().distinct().count()
distinctAirportCount

# COMMAND ----------

# Setting initial page ranks to 10 as specified in assignment directions 
ranks = groupedAirportRDD.keys().distinct().map(lambda x : ((x), 10))
ranks.keys().collect()

# COMMAND ----------

N = distinctAirportCount
teleport = 0.15
dampFactor = (1-teleport)
t = teleport * (1/N) + dampFactor

# Map 1 - For each airport determine page rank contributions and push to connected airports
# Reduce - For every airport get summation of incoming pagerank contributions and update ranking
# Map 2 - Apply teleportation and damping factor
for i in range(pageRankIterations):
  ranks = groupedAirportRDD.join(ranks)\
    .flatMap(lambda x : [(outLink, float(x[1][1])/len(x[1][0])) for outLink in x[1][0]])\
    .reduceByKey(lambda x,y: x+y).mapValues(lambda x : x * t)
  
sortedRanks = ranks.sortBy(lambda x : -x[1]).toDF(["AIRPORT_ID", "PAGE_RANK"])
display(sortedRanks)

# COMMAND ----------

# Write to Amazon S3 Bucket
sortedRanks.coalesce(1).write.csv(airportRankSaveDestination)

# COMMAND ----------

### Part 2 - Tweet Processing & Classification using Pipelines
## Providing the 2 input parameters required for input and output file - files stored in mounted AMZN S3 bucket
tweetInputFile = "/mnt/S3_Assignment2/Tweets.csv" # S3 location to pull input file via mount
tweetOutputFile = "dbfs:/mnt/S3_Assignment2/TweetClassificationMetrics.txt" # S3 location to save output file

df = spark.read.csv(tweetInputFile, header=True, inferSchema= True)

from pyspark.sql.functions import *
df = df.filter(col("text").isNotNull())

from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.evaluation import RegressionEvaluator

# Tokenizer - doing this before StopWordRemover as it expects array of strings as input 
from pyspark.ml.feature import Tokenizer
tokenizeThem = Tokenizer(inputCol="text", outputCol="tokenText")
# Remove stop words from the text column 
from pyspark.ml.feature import StopWordsRemover
removeThem = StopWordsRemover(inputCol=tokenizeThem.getOutputCol(), outputCol="filteredText")
# TERM HASHING
from pyspark.ml.feature import HashingTF, IDF
hashingTF = HashingTF(inputCol =removeThem.getOutputCol(), outputCol = "termFreqRawVector", numFeatures=1e5) #1e5
idf = IDF(inputCol=hashingTF.getOutputCol(), outputCol="textFeatures")

# LABEL CONVERSION
from pyspark.ml.feature import StringIndexer
indexer = StringIndexer(inputCol="airline_sentiment", outputCol="sentimentIndex")


# COMMAND ----------

preprocessPipeline = Pipeline(stages =[tokenizeThem, removeThem, hashingTF, idf, indexer])
processedData = preprocessPipeline.fit(df).transform(df).select("textFeatures", "sentimentIndex").toDF("features", "label")
display(processedData.collect())

# COMMAND ----------

lr = LogisticRegression(maxIter=1000, regParam=0.1, elasticNetParam=0.0, featuresCol = 'features', labelCol = 'label')
paramGrid = ParamGridBuilder()\
.addGrid(lr.maxIter, [10, 100, 1000])\
.addGrid(lr.regParam, [0.0, 0.1, 0.3])\
.addGrid(lr.elasticNetParam, [0.0, 0.5]).build()

# COMMAND ----------

train, test = processedData.randomSplit([0.80, 0.20], 101)

# COMMAND ----------

from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.mllib.evaluation import MulticlassMetrics
crossValidate = CrossValidator(estimator=lr, estimatorParamMaps = paramGrid, evaluator = MulticlassClassificationEvaluator(predictionCol="prediction"), numFolds = 5, parallelism =2)
crossValidatedModel = crossValidate.fit(train)

# COMMAND ----------

predictionCrossValidate = crossValidatedModel.bestModel.transform(test)
display(predictionCrossValidate.collect())

# COMMAND ----------

metrics = MulticlassMetrics(predictionCrossValidate.select(['label', 'prediction']).rdd)

# Overall statistics
accuracy = metrics.accuracy
precision = metrics.precision(0.0)
recall = metrics.recall(0.0)
f1Score = metrics.fMeasure(0.0)
print("Summary Stats")
print("Accuracy = %s" % accuracy)
print("Precision = %s" % precision)
print("Recall = %s" % recall)
print("F1 Score = %s" % f1Score)

# COMMAND ----------

# Writing txt file with evaluation metrics to S3 bucket
dbutils.fs.put(tweetOutputFile, "Evaluation Metrics\n Accuracy = " + str(accuracy) +" \n Precision = " + str(precision) + "\n Recall = " + str(recall) + "\n F1 Score = " + str(f1Score))
