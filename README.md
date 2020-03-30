# CC Project 1

## How to run our code

1. First run cred_transfer.sh to transfer all the aws credentials and necessary scripts from local to raspberry pi and controller (ec2 instance). Note: Change the local directory in the script file before you run.
2. Run panda.py in raspberry pi inside the folder Darknet (This script runs a shell command which uses gnome terminal command to run recorder.py, uploader.py and pi_run.py in raspberry pi at the same time).
3. Make sure the static ec2 instance where the controller is present is running.
