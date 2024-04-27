AWS SETUP :
Let’s Start with setting up AWS environment.
1.	Get AWS learner module access and setup account.
2.	Start AWS Lab and Launch Console.

EMR Cluster SETUP :
1.	Go to services and select “EMR”.
2.	For EMR Configuration, select  
i)	“EMR 5.36.0 “
ii)	Custom (Hadoop 2.10.1, Spark 2.4.8, Zeppelin 0.10.0)
iii)	Setup Cluster Configuration Primary as m4.large, core as m4.large and Tasks as m4.large. Select instance size as 4 for both core and tasks.

3.	Create Two such instances.
4.	Then Create Cluster. Now wait for the status to go from “starting” to “waiting”. 

Network firewall Setup.
1.	Go to Security Firewall and select “Primary”.
2.	Go to Inbound Network Rules and select Edit option. 
3.	Create a new rule to allow “SSH” Connection from myIP.

Connecting SSH Connection
1.	Create a folder and move python files and key pair file inside folder.
2.	Open Terminal and Set the current directory using cd command. Move into the directory with “.pem” and “.py”  files.
Command : cd /user/path/folder
3.	Check key pair file access. 
Command : chmod 400 keypair.pem
4.	Connect to EMR Cluster using below command.
ssh -i /filepath/keypair.pem hadoop@ec2-107-23-222-201. compute-1.amazonaws.com

Publishing python files to EMR Instances
1.	Hosting the training and prediction python codes in EMR 
2.	Connect to EC2 using sftp command:
 sftp -i "keypair.pem" hadoop@ec2-107-23-222-201. compute-1.amazonaws.com
Publish python files using put command:
put /filepath/to/your/pythonfile/filename.py
3.	Connect to EMR using ssh command and verify the files using 
Pwd and ls commands 

Create S3 Bucket 
1. Go to services and select “S3”.
2. Choose a bucket name, and go with default settings
3. Choose Create Bucket
4. Upload both the datasets, training.csv and validation.csv
5. Use this bucket to save the model after running the model through   EMR Cluster.

Create Docker
1.	docker build -t sj764/prediction.
2.	docker push sj764/prediction:latest
3.	docker run -v /home/ec2-user/code/data/csv/:code/data/csv sj764/wine-prediction validation.csv
