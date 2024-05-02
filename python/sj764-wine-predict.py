import os
from pyspark.sql import SparkSession
from pyspark.ml.classification import RandomForestClassificationModel
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

def predict_model(model_path, validation_dataset_path, feature_columns):
    # Create Spark session
    spark = SparkSession.builder \
        .appName("WineQuality") \
        .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem") \
        .config("spark.hadoop.fs.s3a.aws.credentials.provider", "org.apache.hadoop.fs.s3a.TemporaryAWSCredentialsProvider") \
        .getOrCreate()

    # Load the trained model from S3
    model = RandomForestClassificationModel.load(model_path)

    # Read the validation dataset from S3
    validation = spark.read.csv(validation_dataset_path, header=True, inferSchema=True)

    # Assemble features
    va = VectorAssembler(inputCols=feature_columns, outputCol="features")
    validation = va.transform(validation)

    # Make predictions on the validation dataset
    predictions = model.transform(validation)

    # Evaluate the predictions using F1 score
    evaluator = MulticlassClassificationEvaluator(labelCol="quality", predictionCol="prediction", metricName="f1")
    f1_score = evaluator.evaluate(predictions)

    # Print F1 score
    print("F1 Score:", f1_score)

    # Stop Spark session
    spark.stop()

if __name__ == "__main__":
    # S3 paths
    model_s3_path = "s3://your-bucket-name/path-to-your-model"
    validation_dataset_s3_path = "s3://your-bucket-name/path-to-your-validation-dataset"

    # Define feature columns
    feature_columns = ["feature1", "feature2", ...]  # Update with your feature columns

    # Predict using the model and validation dataset
    predict_model(model_s3_path, validation_dataset_s3_path, feature_columns)
