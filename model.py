# # Importing the libraries
# import numpy as np
# import matplotlib.pyplot as plt
# import pandas as pd
# import pickle


# dataset = pd.read_csv('hiring.csv')

# dataset['experience'].fillna(0, inplace=True)

# dataset['test_score'].fillna(dataset['test_score'].mean(), inplace=True)

# X = dataset.iloc[:, :3]

# #Converting words to integer values
# def convert_to_int(word):
#     word_dict = {'one':1, 'two':2, 'three':3, 'four':4, 'five':5, 'six':6, 'seven':7, 'eight':8,
#                 'nine':9, 'ten':10, 'eleven':11, 'twelve':12, 'zero':0, 0: 0}
#     return word_dict[word]

# X['experience'] = X['experience'].apply(lambda x : convert_to_int(x))

# y = dataset.iloc[:, -1]

# #Splitting Training and Test Set
# #Since we have a very small dataset, we will train our model with all availabe data.

# from sklearn.linear_model import LinearRegression
# regressor = LinearRegression()

# #Fitting model with trainig data
# regressor.fit(X, y)

# # Saving model to disk
# pickle.dump(regressor, open('model.pkl','wb'))

# # Loading model to compare the results
# model = pickle.load(open('model.pkl','rb'))
# print(model.predict([[2, 9, 6]]))

########################################################################


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle

# Importing the dataset
data = pd.read_csv('cpdata.csv')

label= pd.get_dummies(data.label).iloc[: , 1:]
data= pd.concat([data,label],axis=1)
data.drop('label', axis=1,inplace=True)
print('The data present in one row of the dataset is')
print(data.head(1))
train=data.iloc[:, 0:4].values
test=data.iloc[: ,4:].values  



# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(train, test, test_size = 0.2, random_state = 0)
#divide the test data set as 0.2*10000=2000 observation remaining for training i.e. 8000
print(X_train)
print(X_test)
# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
#transform finds actual value then normalize it
#fit_transform does both steps at once

##############################################################################################################
#ANN

# Importing the Keras libraries and packages
import keras
from keras.models import Sequential
from keras.layers import Dense

#=============================================================================
# Initialising the layers for ANN
classifier = Sequential()
#hidden layer -> neuron
#here we are building a double hidden layer ANN module 
#no of neurons in hidden layer is decided by (no. of input nodes + no. of output nodes)/2 = (4+1)/2=2
# Adding the input layer and the first hidden layer
classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = 4))
#units -> no. of neurons
#kernel_initializer -> assigning weights 
#activation -> activation function ,for Rectifier function its "relu" 
#input_dim -> no. of input nodes

# Adding the second hidden layer
classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))

classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))
#classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))


# Adding the output layer
classifier.add(Dense(units = 30, kernel_initializer = 'uniform', activation = 'sigmoid'))


# Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
# optimizer -> algorithm u want to use to find optimal set of weights and "adam" is an algorithm them
# loss      -> loss function ,since output is binary form hence binary_crossentropy
# metrics   -> 
# Fitting the ANN to the Training set
classifier.summary()
classifier.fit(X_train, y_train, batch_size = 10, epochs = 100,verbose=1, validation_data=(X_test, y_test))
score = classifier.evaluate(X_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

# Saving model to disk
pickle.dump(classifier, open('model.pkl','wb'))
# Loading model to compare the results
model = pickle.load(open('model.pkl','rb'))

l=[]
l.append(30)
l.append(92)
l.append(6.05)
l.append(116)
predictcrop=[l]


# Putting the names of crop in a single list
crops=['APPLE','BANANA','BLACK GRAM','CHICKPEA','COCONUT','COFFEE','COTTON','GRAPES','GROUNDNUT','JUTE','KIDNEY BEANS','LENTIL','MAIZE','MANGO','MILLET','MOTH BEANS','MUNG BEAN','MUSK MELON','ORANGE','PAPAYA','PEAS','PIGEON PEAS','POMEGRANATE','RICE','RUBBER','SUGARCANE','TEA','TOBACCO','WATERMELON','WHEAT']
cr='rice'

##########################################################################################
#Predicting the crop
fi=sc.transform(np.array(predictcrop))
print(fi)
predictions = model.predict(fi)



##########################################################################################
#=====================================================================================
#get the position of predicted_crops in crops list
import copy
prediction1 = copy.copy(predictions)
prediction1.sort()
max=[prediction1[0][29],prediction1[0][28],prediction1[0][27],prediction1[0][26],prediction1[0][25]]
pos=[-1]*5
for k in range (0,5):
  for j in range(0,30):
    if(max[k]==predictions[0][j] and max[k]>=0.100000):
     pos[k]=j

######################################################################################################
#========================================================================================== 
#Display Predicted crops     
print('\n\n')
print('======================================================================================')  
print('Predicted crops are')

for i in range(0,5):
  if(pos[i]!=-1):
    print(crops[pos[i]])
  ####################################################################