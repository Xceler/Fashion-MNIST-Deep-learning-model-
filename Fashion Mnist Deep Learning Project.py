import numpy as np 
import pandas as pd 
import tensorflow as tf 
from sklearn.preprocessing import LabelEncoder 
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, BatchNormalization
from tensorflow.keras.models import Sequential 
from tensorflow.keras.utils import to_categorical 

#Data Handling For train and test data 

train_data = pd.read_csv('fashion-mnist_train.csv')
test_data = pd.read_csv('fashion-mnist_test.csv')


x_train = train_data.drop(columns = 'label').values 
y_train = train_data['label'].values 

x_test = test_data.drop(columns = 'label').values 
y_test = test_data['label'].values 


#Reshaping X_train and X_test 

img_size = 28
num_color = 1

x_train = x_train.reshape(-1, img_size, img_size, num_color)
x_train_data = x_train.astype('float32')/255.0

x_test = x_test.reshape(-1, img_size, img_size, num_color)
x_test_data = x_test.astype('float32')/255.0

#LabelEncoding 

label_encode = LabelEncoder()

y_train = label_encode.fit_transform(y_train)
y_train_data = to_categorical(y_train)

y_test = label_encode.fit_transform(y_test)
y_test_data = to_categorical(y_test)


#Building and Developing Deep Learning Architecture 

model  = Sequential([

    Conv2D(32, (3,3), activation = 'relu', padding = 'same', input_shape = (img_size, img_size, num_color)),
    MaxPooling2D((2,2)),
    BatchNormalization(),

    Conv2D(64, (3,3), activation = 'relu', padding = 'same'),
    MaxPooling2D((2,2)), 
    BatchNormalization(),

    Conv2D(64, (3,3), activation  = 'relu', padding = 'same'),
    MaxPooling2D((2,2)),
    BatchNormalization(),

    Flatten(),
    
    Dense(1024, activation = 'relu'), 
    BatchNormalization(),

    Dense(1024, activation = 'relu'),
    BatchNormalization(),

    Dense(10, activation = 'softmax')
])


#Compiling the model 

model.compile(loss = 'categorical_crossentropy',
              optimizer = 'adam', 
              metrics = ['accuracy'])


#Fit the model 
model.fit(x_train_data, y_train_data, epochs = 10, validation_data = (x_test_data, y_test_data))