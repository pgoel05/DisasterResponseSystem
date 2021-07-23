# Disaster Response System

### Table of Contents

1. [Installation](#installation)
2. [Project Motivation](#motivation)
3. [Project Description](#description)
4. [File Descriptions](#files)
5. [Instructions](#instructions)
6. [Licensing](#licensing)

## Installation <a name="installation"></a>

Python 3 is the main requirement of the project along with the below mentioned libraries:
- numpy 
- pandas
- sqlalchemy
- re
- nltk
- pickle
- sklearn
- plotly
- flask

## Project Motivation<a name="motivation"></a>

Completing this project helped me dive deep into the following domains:
- ETL (Extract, Transform, Load)
- Multi-label classification
- Machine Learning Pipelines

## Project Description <a name="description"></a>

This system is a multi-label classification based approach which aims at classifying the messages that are sent during disasters. By classifying these messages, we can allow these messages to be sent to the appropriate disaster relief agency. It starts with a ETL pipeline followed by a Machine Learning pipeline. Finally, a web app is deployed using flask framework where you can input a message and get classification results.

The data set is provided by [Figure Eight](https://www.figure-eight.com/) containing real messages that were sent during disaster events.

## File Descriptions <a name="files"></a>

        Disaster_Response_System
          |-- app
                |-- templates
                        |-- go.html
                        |-- master.html
                |-- run.py
          |-- data
                |-- disaster_message.csv
                |-- disaster_categories.csv
                |-- disaster_data.db
                |-- process_data.py
          |-- models
                |-- classifier.pkl
                |-- train_classifier.py
          |-- README

1. App folder contains files for the web application.
    - "go.html", "master.html" : Template files
    - "run.py" : Flask application file
2. Data folder contains data files along with the database and the ETL file.
    - disaster_message.csv : Features file
    - disaster_categories.csv : Labels file
    - disaster_data.db : Database file
    - process_data.py : ETL file
3. Models folder contains the Machine Learning model.
    - classifier.pkl : Model pickle file
    - train_classifier.py : Machine Learning pipeline file

## Instructions <a name="instructions"></a>

1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/disaster_data.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/disaster_data.db models/classifier.pkl
        `
2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

## Licensing <a name="licensing"></a>

Feel free to use the above code as you would like! 