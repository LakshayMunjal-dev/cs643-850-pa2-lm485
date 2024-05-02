# Docker Hub:
https://hub.docker.com/repository/docker/sj764/programmingassignment2/general

# AWS SETUP :
Let’s Start with setting up AWS environment.
1. Get AWS learner module access and setup account.
2. Start AWS Lab and Launch Console.
# EMR Cluster SETUP :
1. Go to services and select “EMR”.
2. For EMR Configuration, select:
i) “EMR 5.36.0 “
ii) Custom (Hadoop 2.10.1, Spark 2.4.8, Zeppelin 0.10.0)
iii) Setup Cluster Configuration Primary as m4.large, core as
m4.large and Tasks as m4.large. Select instance size as 4 for
both core and tasks.
3. Create Two such instances.
4. Then Create Cluster. Now wait for the status to go from “starting”
to “waiting”.

# Network firewall Setup.
1. Go to Security Firewall and select “Primary”.
2. Go to Inbound Network Rules and select Edit option.
3. Create a new rule to allow “SSH” Connection from my IP.
   
# Connecting SSH Connection
1. Create a folder and move python files and key pair file inside
folder.
2. Open Terminal and Set the current directory using cd command.
Move into the directory with “.pem” and “.py” files.
Command : cd /user/path/folder
3. Check key pair file access.
Command : chmod 400 keypair.pem
4. Connect to EMR Cluster using below command.
ssh -i /filepath/keypair.pem hadoop@ec2-107-23-222-201.
compute-1.amazonaws.com
ssh -i CloudPA2.pem hadoop@ec2-54-156-33-31.compute-
1.amazonaws.com

# Create S3 Bucket
1. Go to services and select “S3”.
2. Choose a bucket name, and go with default settings
3. Choose Create Bucket
4. Upload both the datasets, training.csv and validation.csv
5. Upload both python files, training and prediction as well as docker file
to bucket.
6. Use this bucket to save the model after running the model through
EMR Cluster.

# Publishing python files to EMR Instances
1. Hosting the training and prediction python codes in EMR
2. Sync files to EMR primary instance with AWS S3 bucket using
command: aws s3 sync s3://sj764pa2/ .
3. verify the files using ls commands.
   
# Create Docker
1. Create an account on Docker Hub and save credentials.
2. Now in terminal use command: docker login
3. Then use this command to build a repository:
4. docker build -t sj764/programmingassignment2 .
5. Run prediction in docker: docker run sj764/programmingassignment2.
6. Then push you image to docker container:
docker push sj764/programmingassignment2 .

# Train ML model in Spark cluster with 4 ec2 instances in parallel
1. Perform SSH to Master of cluster using below command:
ssh -i "ec2key.pem" <<User>>@<<Public IPv4 DNS>>
2. On successful login to master , change to root user by running
command: sudo su
3. Submit job using following command:
Run training: spark-submit sj764-wine-train.py.
4. You can trace status of this job in EMR UI application logs.Once
status is succeeded, you can see trainingmodel.model created
within s3 bucket.
5. After that re-sync s3 bucket with cluster to read model and run
prediction code using: spark-submit sj764-wine-predict.py.
