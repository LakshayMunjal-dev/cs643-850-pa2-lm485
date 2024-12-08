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

    # Read the validation dataset from S3 (adjusting for semicolon delimiter)
    validation = spark.read.csv(validation_dataset_path, header=True, inferSchema=True, sep=';')

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
    # S3 paths (updated with actual model and dataset locations)
    model_s3_path = "s3://lm485-bucket/models/trainingmodel.model"  # Path to the trained model in 'models' folder
    validation_dataset_s3_path = "s3://lm485-bucket/data/ValidationDataset.csv"  # Path to the validation dataset in 'data' folder

    # Define feature columns (exclude 'quality' column from features)
    feature_columns = [
        "fixed acidity", "volatile acidity", "citric acid", "residual sugar", "chlorides",
        "free sulfur dioxide", "total sulfur dioxide", "density", "pH", "sulphates", "alcohol"
    ]

    # Predict using the model and validation dataset
    predict_model(model_s3_path, validation_dataset_s3_path, feature_columns)
