# Disaster Response Pipeline Project
## Installation <a name="installation"></a>
Here are the list of libraries needed to use run this project:
sklearn, pandas, numpy, sql_alchemy, pickle, nltk, textblob

## Brief summary<a name="motivation"></a>
This project is a part of collaborative effort with figure-8 to classify messages sent during disasters with machine learning model.


## Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `cd ./data`
        
        `python ./process_data.py`
    - To run ML pipeline that trains classifier and saves
        `cd ./models`
        
        `python ./train_classifier.py`

2. Go to `app` directory: `cd app`

3. Run your web app: `python run.py`

4. The command prompt will provide you the link to the page which look like this (http://0.0.0.0:3000/)

<img src='Page_pic.jfif'></img>

## Licensing, Authors, Acknowledgements
Thanks to Figure-8 for providing the data for this project.


You can the code to this project via: xxx
