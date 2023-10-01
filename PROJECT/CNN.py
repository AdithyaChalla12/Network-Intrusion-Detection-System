#Make sure all the following libraries are installed before running the application. 
import warnings
import pandas as pd
import numpy as np
import sklearn
from sklearn.utils import shuffle
from sklearn.metrics import *
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import warnings

from keras.preprocessing import sequence
from tensorflow.keras import optimizers
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Embedding, SimpleRNN, BatchNormalization
from keras.models import model_from_json

warnings.filterwarnings("ignore")
print("LOADING TRAINING AND TESTING DATA")
#load the csv file containing the column names 
column_name = pd.read_csv("Field Names.csv", header = None)
#Convert the array into list
new_columns = list(column_name[0].values)
#adding difficulty 
new_columns += ['class', 'difficulty']
#loading train and test data files
train_data = pd.read_csv('CSE-CIC-IDS2017Train+.txt', names = new_columns)
test_data = pd.read_csv('CSE-CIC-IDS2017Test+.txt', names = new_columns)
#Training data sample
print("The training data is")
train_data.tail()
#Output total rows and columns of dataframe
print(f"The shape of the training dataframe is : {train_data.shape}")
#Same for testing
print("The testing data is")
test_data.head()
#Idem dito ^
print(f"The shape of the testing dataframe is : {test_data.shape}")
#Load attacks.txt containing the attack categories
map_attacks = [x.strip().split() for x in open('attacks.txt', 'r')]
map_attacks = {k:v for (k,v) in map_attacks}
#Replace the "class" column values to 5 attack categories in training and testing dataframe
train_data['class'] = train_data['class'].replace(map_attacks)
test_data['class'] = test_data['class'].replace(map_attacks)
train_data = shuffle(train_data)

print("DATA PREPROCESSING")
#separate the training dataframe into feature columns and label columns
X = train_data.drop('class', axis = 1) #Independent features
y = train_data['class'] #Dependent features (Labels)
#Converting String to Integer with get_dummies by pandas
columns = ['protocol_type', 'service', 'flag']
X_new = pd.get_dummies(X, columns = columns, drop_first = True)
#Idem dito for class ^
y_new = train_data['class']
y_new = pd.get_dummies(y_new)
#Split data: 80% training and 20% testing 
X_train, X_test, y_train, y_test = train_test_split(X_new, y_new, test_size = 0.2, random_state = 101)
#Use StandardScaler() to standardize data - explained in Honours Project
sc = StandardScaler()
sc.fit(np.array(X_train))
X_train = sc.transform(X_train)
X_test = sc.transform(X_test)
#Use the keras's sequential API 
#First dense layer takes an input parameter as 256 (number of neurons in the first layer).
#The second parameter "input_dim" corresponds to the input features. 
#Use "relu" as activation function.
#The activation function for last dense layer is "softmax" because of the multiple classes, further explained in document.
#Set dropout for 10%. 
model1 = Sequential()
model1.add(Dense(64, input_dim = 120, activation = "relu", kernel_initializer = "lecun_normal"))
model1.add(Dense(128, activation = "relu"))
model1.add(Dense(5, activation = "softmax"))
#Summary of model architecture listing information about parameters per layer. 
model1.summary()
#Three paramaters: 
#Loss - The loss function.
#Optimizer - To minimize the loss function.
#Metrics - The mode of evaluation for our model.
#"categorical_loss" - is used because of the multi-class classifcation problem.
#"adam" - The updated version of SGD.
optim = optimizers.SGD(lr = 0.0001)
model1.compile(loss = 'categorical_crossentropy', optimizer = optim, metrics = ['accuracy'])
#Fit the model on our data.
#X_train - The feature columns of the training data.
#y_train - The labels columns of the training data.
#validation_data - The validation data
#batch_size and epochs further explained in document. 
history = model1.fit(X_train, y_train, 
          validation_data = (X_test, y_test),
          batch_size = 32, 
          epochs = 20)
#use matplitlib to draw the plots
plt.figure(figsize = (15, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label = "TRAINING LOSS")
plt.plot(history.history['val_loss'], label = "VALIDATION LOSS")
plt.title("TRAINING LOSS vs VALIDATION LOSS")
plt.xlabel("EPOCH'S")
plt.ylabel("TRAINING LOSS vs VALIDATION LOSS")
plt.legend(loc = "best")

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label = "TRAINING ACCURACY")
plt.plot(history.history['val_accuracy'], label = "VALIDATION ACCURACY")
plt.title("TRAINING ACCURACY vs VALIDATION ACCURACY")
plt.xlabel("EPOCH'S")
plt.ylabel("TRAINING ACC vs VALIDATION ACCURACY")
plt.legend(loc = "best")
#Serialize model 1 , save with json. 
model_json = model1.to_json() 
with open("model1.json", "w") as json_file:
    json_file.write(model_json)
model1.save_weights('model1_weights.h5')
print("Saved model to disk")
# load model 1. 
json_file = open("model1.json", "r") 
loaded_model_json = json_file.read() 
json_file.close() 
loaded_model = model_from_json(loaded_model_json) 
loaded_model.load_weights("model1_weights.h5")
print("Loaded model from disk")
