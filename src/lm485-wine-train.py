import findspark
findspark.init()
findspark.find()

# Load libraries
import pyspark
from pyspark.mllib.tree import RandomForest
from pyspark.mllib.regression import LabeledPoint
from pyspark.ml.feature import VectorAssembler
from pyspark.sql import SparkSession
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, classification_report, confusion_matrix, accuracy_score

# Start Spark session
conf = pyspark.SparkConf().setAppName('winequality').setMaster('local')
sc = pyspark.SparkContext(conf=conf)
spark = SparkSession(sc)

# Load the dataset
df = spark.read.format("csv").load("s3://lm485-bucket/data/TrainingDataset.csv", header=True, sep=";")
df.printSchema()
df.show()

# Rename and cast columns
df = df.withColumnRenamed('""""quality"""""', "label")  # Remove extra quotes around 'quality'
for col_name in df.columns[:-1]:  # Loop through columns excluding the last (which is 'label')
    df = df.withColumn(col_name, df[col_name].cast('float'))

# Assemble features
feature_columns = df.columns[:-1]  # Exclude the 'label' column
vectorAssembler = VectorAssembler(inputCols=feature_columns, outputCol='features')
df_tr = vectorAssembler.transform(df).select(['features', 'label'])

# Convert to RDD of LabeledPoints
dataset = df_tr.rdd.map(lambda row: LabeledPoint(row['label'], row['features']))

# Split the dataset
training, test = dataset.randomSplit([0.7, 0.3], seed=11)

# Create the Random Forest model
RFmodel = RandomForest.trainClassifier(training, numClasses=10, categoricalFeaturesInfo={},
                                       numTrees=21, featureSubsetStrategy="auto", impurity='gini', 
                                       maxDepth=30, maxBins=32)

# Predictions
predictions = RFmodel.predict(test.map(lambda x: x.features))
labelsAndPredictions = test.map(lambda lp: lp.label).zip(predictions)

# Convert to DataFrame
labelsAndPredictions_df = labelsAndPredictions.toDF(["label", "Prediction"])
labelsAndPredictions_df.show()

# Convert to Pandas DataFrame for metrics calculation
labelpred_df = labelsAndPredictions_df.toPandas()

# Calculate metrics
F1score = f1_score(labelpred_df['label'], labelpred_df['Prediction'], average='micro')
print("F1- score: ", F1score)
print(confusion_matrix(labelpred_df['label'], labelpred_df['Prediction']))
print(classification_report(labelpred_df['label'], labelpred_df['Prediction']))
print("Accuracy: ", accuracy_score(labelpred_df['label'], labelpred_df['Prediction']))

# Calculate test error
testErr = labelsAndPredictions.filter(lambda lp: lp[0] != lp[1]).count() / float(test.count())
print('Test Error = ' + str(testErr))

# Save model
RFmodel.save(sc, 's3://lm485-bucket/models/trainingmodel.model/')
