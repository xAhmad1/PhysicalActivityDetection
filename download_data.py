import io
import requests
import zipfile
import shutil
import os
import pandas as pd
import warnings
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=pd.errors.SettingWithCopyWarning)
def dataCleaning(dataCollection):
    # removing all entries with activity labeled as 0 (transient activity)
    dataCollection = dataCollection.drop(dataCollection[dataCollection.activityID == 0].index)
    # filling NaN values using an interpolation method
    dataCollection = dataCollection.interpolate()
    return dataCollection

CHALLENGE_NAME = "Physical_Activity_Prediction"

# Download the data from Dropbox
url = "https://www.dropbox.com/s/2z4q1detgm18jhp/Protocol.zip?dl=1"  # Updated URL to dl=1 for direct download
filename = "Protocol.zip"
response = requests.get(url, stream=True)

print("Downloading data...")
with open(filename, "wb+") as f:
    f.write(response.content)

# Extract the contents of the zip file
with zipfile.ZipFile(filename, "r") as zip_ref:
    zip_ref.extractall()

print("Extracting data...")

# Removing the zip file
os.remove(filename)


activityIDdict = {0: 'transient',
                  1: 'lying',
                  2: 'sitting',
                  3: 'standing',
                  4: 'walking',
                  5: 'running',
                  6: 'cycling',
                  7: 'Nordic_walking',
                  9: 'watching_TV',
                  10: 'computer_work',
                  11: 'car driving',
                  12: 'ascending_stairs',
                  13: 'descending_stairs',
                  16: 'vacuum_cleaning',
                  17: 'ironing',
                  18: 'folding_laundry',
                  19: 'house_cleaning',
                  20: 'playing_soccer',
                  24: 'rope_jumping' }

colNames = ["timestamp", "activityID","heartrate"]

IMUhand = ['handTemperature', 
           'handAcc16_1', 'handAcc16_2', 'handAcc16_3', 
           'handAcc6_1', 'handAcc6_2', 'handAcc6_3', 
           'handGyro1', 'handGyro2', 'handGyro3', 
           'handMagne1', 'handMagne2', 'handMagne3',
           'handOrientation1', 'handOrientation2', 'handOrientation3', 'handOrientation4']

IMUchest = ['chestTemperature', 
           'chestAcc16_1', 'chestAcc16_2', 'chestAcc16_3', 
           'chestAcc6_1', 'chestAcc6_2', 'chestAcc6_3', 
           'chestGyro1', 'chestGyro2', 'chestGyro3', 
           'chestMagne1', 'chestMagne2', 'chestMagne3',
           'chestOrientation1', 'chestOrientation2', 'chestOrientation3', 'chestOrientation4']

IMUankle = ['ankleTemperature', 
           'ankleAcc16_1', 'ankleAcc16_2', 'ankleAcc16_3', 
           'ankleAcc6_1', 'ankleAcc6_2', 'ankleAcc6_3', 
           'ankleGyro1', 'ankleGyro2', 'ankleGyro3', 
           'ankleMagne1', 'ankleMagne2', 'ankleMagne3',
           'ankleOrientation1', 'ankleOrientation2', 'ankleOrientation3', 'ankleOrientation4']

columns = colNames + IMUhand + IMUchest + IMUankle
subjectID = [1,2,3,4,5,6,7,8,9]

dataCollection = pd.DataFrame()
directory = 'Protocol'
# iterate over files in that directory
for filename in os.listdir(directory):
    f = os.path.join(directory, filename)
    if os.path.isfile(f): 
        procData = pd.read_table(f, header=None, sep='\s+')
        procData.columns = columns
        procData['subject_id'] = int(f[-5])
        dataCollection = dataCollection.append(procData, ignore_index=True)

dataCollection.reset_index(drop=True, inplace=True)
dataCol = dataCleaning(dataCollection)
dataCol["heartrate"].iloc[:4]=100

public_train_subjects = [1, 2, 3, 4, 5, 9]
public_test_subjects = [6, 7]
private_subjects = [8]
public_train = dataCol[dataCol['subject_id'].isin(public_train_subjects)]
private = dataCol[dataCol['subject_id'].isin(private_subjects)]
public_test = dataCol[dataCol['subject_id'].isin(public_test_subjects)]
public_data_train = public_train.drop(columns = ['activityID'])
public_labels_train = public_train['activityID']
public_data_test = public_test.drop(columns = ['activityID'])
public_labels_test = public_test['activityID']
private_data = private.drop(columns = ['activityID'])
private_labels = private['activityID']

public_path = "./data/public/"
private_path = "./data/private/"
if os.path.exists("./data"):
    shutil.rmtree("./data")
os.makedirs(public_path)
os.makedirs(private_path)

public_data_train.to_csv(os.path.join(public_path, "data_train.csv"),
                    index=False)
public_labels_train.to_csv(os.path.join(public_path, "labels_train.csv"),
                    index=False)

public_data_test.to_csv(os.path.join(public_path, "data_test.csv"),
                    index=False)
public_labels_test.to_csv(os.path.join(public_path, "labels_test.csv"),
                    index=False)

private_data.to_csv(os.path.join(private_path, "data_priv.csv"),
                    index=False)
private_labels.to_csv(os.path.join(private_path, "labels_priv.csv"),
                    index=False)
                  
shutil.rmtree("./Protocol")
print("Done")