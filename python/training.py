# Import necessary libraries
from pyspark.sql.functions import col, isnan
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier, DecisionTreeClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml import Pipeline

# Create a Spark session
spark = SparkSession.builder\
    .master("local")\
    .appName("CS643_Wine_Quality_Predictions_Project")\
    .config("spark.jars.packages", "org.apache.hadoop:hadoop-aws:3.2.2")\
    .getOrCreate()

# Configure Spark to work with AWS S3
# (Replace access and secret keys with your actual AWS credentials)
spark.sparkContext._jsc.hadoopConfiguration().set("com.amazonaws.services.s3.enableV4", "true")
spark.sparkContext._jsc.hadoopConfiguration().set("fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")
spark.sparkContext._jsc.hadoopConfiguration().set("fs.s3a.aws.credentials.provider", "com.amazonaws.auth.InstanceProfileCredentialsProvider,com.amazonaws.auth.DefaultAWSCredentialsProviderChain")
spark.sparkContext._jsc.hadoopConfiguration().set("fs.AbstractFileSystem.s3a.impl", "org.apache.hadoop.fs.s3a.S3A")
spark.sparkContext._jsc.hadoopConfiguration().set("fs.s3a.access.key","ASIARNFNO5U42XKSFTAB")
spark.sparkContext._jsc.hadoopConfiguration().set("fs.s3a.secret.key","yIv1SEeUsNJ9DBcHd8VYU+A7TLDaGYBAk9b1SxsX")
spark.sparkContext._jsc.hadoopConfiguration().set("fs.s3a.endpoint", "s3.us-east-1.amazonaws.com")

# Load training and validation datasets from S3
df = spark.read.format("csv")\
    .option("header", "true")\
    .option("inferSchema", "true")\
    .option("sep", ";")\
    .load("s3://sj764pa2/TrainingDataset.csv")

validation_df = spark.read.format("csv")\
    .option("header", "true")\
    .option("inferSchema", "true")\
    .option("sep", ";")\
    .load("s3://sj764pa2/ValidationDataset.csv")

# Check if data is loaded successfully
if df.count() > 0 and validation_df.count() > 0:
    print("Data loaded successfully")
else:
    print("Something unexpected happened during data load")

# Rename columns for consistency
new_column_names = {
    '"""""fixed acidity""""': 'fixed_acidity',
    '"""fixed acidity""""': 'fixed_acidity',
    '""""volatile acidity""""': 'volatile_acidity',
    '""""citric acid""""': 'citric_acid',
    '""""residual sugar""""': 'residual_sugar',
    '""""chlorides""""': 'chlorides',
    '""""free sulfur dioxide""""': 'free_sulfur_dioxide',
    '""""total sulfur dioxide""""': 'total_sulfur_dioxide',
    '""""density""""': 'density',
    '""""pH""""': 'pH',
    '""""sulphates""""': 'sulphates',
    '""""alcohol""""': 'alcohol',
    '""""quality"""""': 'label'
}

for current_name, new_name in new_column_names.items():
    df = df.withColumnRenamed(current_name, new_name)
    validation_df = validation_df.withColumnRenamed(current_name, new_name)

# Display column names
print(df.columns)
print(validation_df.columns)

# Check for null or NaN values in columns
null_counts = []
for col_name in df.columns:
    null_count = df.filter(col(col_name).isNull() | isnan(col(col_name))).count()
    null_counts.append((col_name, null_count))

for col_name, null_count in null_counts:
    print(f"Column '{col_name}' has {null_count} null or NaN values.")

# Split the dataset into training and testing sets
df, test_df = df.randomSplit([0.7, 0.3], seed=42)

# Define data preprocessing and modeling pipeline
assembler = VectorAssembler(
    inputCols=['fixed_acidity', 'volatile_acidity', 'citric_acid', 'residual_sugar', 'chlorides',
               'free_sulfur_dioxide', 'total_sulfur_dioxide', 'density', 'pH', 'sulphates', 'alcohol'],
    outputCol="inputFeatures")

scaler = StandardScaler(inputCol="inputFeatures", outputCol="features")

# Initialize classification models
lr = LogisticRegression()
rf = RandomForestClassifier()
dt = DecisionTreeClassifier(labelCol="label", featuresCol="features", seed=0)

# Create pipelines with different classifiers
pipeline1 = Pipeline(stages=[assembler, scaler, lr])
pipeline2 = Pipeline(stages=[assembler, scaler, rf])
pipeline3 = Pipeline(stages=[assembler, scaler, dt])

# Build a parameter grid for hyperparameter tuning
paramgrid = ParamGridBuilder().build()

# Initialize evaluator for model performance evaluation
evaluator = MulticlassClassificationEvaluator(metricName="f1")

# Cross-validate each pipeline using the specified classifiers
crossval = CrossValidator(estimator=pipeline1, estimatorParamMaps=paramgrid, evaluator=evaluator, numFolds=10)
cvModel1 = crossval.fit(df)
print("F1 Score for LogisticRegression Model: ", evaluator.evaluate(cvModel1.transform(test_df)))

crossval = CrossValidator(estimator=pipeline2, estimatorParamMaps=paramgrid, evaluator=evaluator, numFolds=10)
cvModel2 = crossval.fit(df)
print("F1 Score for RandomForestClassifier Model: ", evaluator.evaluate(cvModel2.transform(test_df)))

crossval = CrossValidator(estimator=pipeline3, estimatorParamMaps=paramgrid, evaluator=evaluator, numFolds=10)
cvModel3 = crossval.fit(df)
print("F1 Score for DecisionTreeClassifier Model: ", evaluator.evaluate(cvModel3.transform(test_df)))

# Save models to S3
model_path = "s3://sj764pa2/LogisticRegression"
cvModel1.save(model_path)

model_path = "s3://sj764pa2/RandomForestClassifier"
cvModel2.save(model_path)

model_path = "s3://sj764pa2/DecisionTreeClassifier"
cvModel3.save(model_path)

# Stop the Spark session
spark.stop()
