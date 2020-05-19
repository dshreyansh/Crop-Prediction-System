# Data Preprocessing
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#pandas->data manipulation and analysis

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
classifier.fit(X_train, y_train, batch_size = 10, epochs = 100)

#Using firebase to import data to be tested
#from firebase import firebase
#firebase =firebase.FirebaseApplication('https://cropit-eb156.firebaseio.com/')
#firebase =firebase.FirebaseApplication('https://cropit-eb156-7d238.firebaseio.com/')
# tp=firebase.get('/Realtime',None)
# atemp=tp['Air Temp']
# ah=tp['Air Humidity']
# #shum=tp['Soil Humidity']
# pH=tp['Soil pH']
# rain=tp['Rainfall']
# District_Name=tp['State Name']
# State_name=tp['District Name']


# l=[]
# l.append(atemp)
# l.append(ah)
# l.append(pH)
# l.append(rain)
# predictcrop=[l]

l=[]
l.append(25)
l.append(80)
l.append(5.2)
l.append(100)
predictcrop=[l]

# Putting the names of crop in a single list
crops=['APPLE','BANANA','BLACK GRAM','CHICKPEA','COCONUT','COFFEE','COTTON','GRAPES','GROUNDNUT','JUTE','KIDNEY BEANS','LENTIL','MAIZE','MANGO','MILLET','MOTH BEANS','MUNG BEAN','MUSK MELON','ORANGE','PAPAYA','PEAS','PIGEON PEAS','POMEGRANATE','RICE','RUBBER','SUGARCANE','TEA','TOBACCO','WATERMELON','WHEAT']
cr='rice'

##########################################################################################
#Predicting the crop
predictions = classifier.predict(sc.transform(np.array(predictcrop)))
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
sms_crops='Predicted crops are:'
for i in range(0,5):
  if(pos[i]!=-1):
    print(crops[pos[i]])
  ####################################################################
#Sending the predicted crop to database
  if i==0:
    if(pos[i]!=-1):
      temp=crops[pos[i]]
      #firebase.put('/croppredicted','Crop1',temp.capitalize())
      sms_crops+=temp.capitalize()+','
  if i==1:
    if(pos[i]!=-1):
      temp=crops[pos[i]]
      #firebase.put('/croppredicted','Crop2',temp.capitalize())
      sms_crops+=temp.capitalize()+','
  if i==2:
    if(pos[i]!=-1):
      temp=crops[pos[i]]
      #firebase.put('/croppredicted','Crop3',temp.capitalize())
      sms_crops+=temp.capitalize()+','
  if i==3:
    if(pos[i]!=-1):
      temp=crops[pos[i]]
      #firebase.put('/croppredicted','Crop4',temp.capitalize())
      sms_crops+=temp.capitalize()+','
  if i==4:
    if(pos[i]!=-1):
      temp=crops[pos[i]]
      #firebase.put('/croppredicted','Crop5',temp.capitalize())
      sms_crops+=temp.capitalize()+','
print('======================================================================================')  
print('\n\n')    
    
#Default crop is rice  
if not max:
  print('default crop is {}'.cr)
  #firebase.put('/croppredicted','Crop1',crops[pos[0]])
  
###########################################################################################################
##############################################################################################################
##############################################################################################################
#TENURE
print('TENURE')
tenure=pd.read_csv('tenure.csv')
tenure_you_want=64 #months 
il=0
str_crop=[]
while il<5:
  if(pos[il]!=-1):
    crop=crops[pos[il]]
    #crop=crop.capitalize()
    print(crop)
    str_crop.append(crop)
    #get the cost per hectar for predicted crop
    tenuredata=tenure[tenure.Crops.str.contains(crop)]
    isempty=tenuredata.empty
    print('Time period required for cultivation is ',float(tenuredata['min.time']),'to',float(tenuredata['max.time']),'months')
    
    temp='Time period required for cultivation is '+str(float(tenuredata['min.time']))+' to '+str(float(tenuredata['max.time']))+' months'
    str_crop.append(temp)
    print('provided tenure:',tenure_you_want,'months')
    temp='provided tenure: '+str(tenure_you_want)+' months'
    str_crop.append(temp)
      
    if isempty==False:
      if ((float(tenuredata['min.time'])<=tenure_you_want) and (tenure_you_want<=float(tenuredata['max.time']) or tenure_you_want>=float(tenuredata['max.time']))):
        print(crop,'can be grown in provided tenure')
        temp=crop+' can be grown in provided tenure'
        str_crop.append(temp)
      else:
        print(crop,'cannot be grown in period you want')
        temp=crop+' cannot be grown in period you want'
        str_crop.append(temp)
        pos[il]=-1
  
  print('===================================================================================================')     
  print('\n\n')
  il+=1
###########################################################################################################
##############################################################################################################
##############################################################################################################
####################################################################################################################   
print("Crops that match to the provided tenure are:")  
il=0
count=0
newlist=[]
while il<5:
  if pos[il]!=-1:
    crop=crops[pos[il]]
    print(crop)
    newlist.append(crop)
    count+=1
  il+=1  

if(count==0):
  print('No crop can be cultivated in tenure given by you')   
print('\n\n')
print('==================================================================================================')
###########################################################################################################
##############################################################################################################
##############################################################################################################
#==================================================================================================
#product Statistics
print('STATISTICS')
loc=[]
loc.append('Maharashtra')
loc.append('buldhana')  
location=[loc]

district_name=location[0][1].upper()
State_Name=location[0][0]

data2=pd.read_csv('apy.csv')  
il=0

while il<len(newlist):
  crop=newlist[il]
  crop=crop.capitalize()
  print(crop)
  data3=data2[data2.Crop.str.contains(crop) & data2.District_Name.str.contains(district_name)&data2.State_Name.str.contains(State_Name)] 
  isempty = data3.empty
  if isempty==False:
    production_year_wise=data3[['Crop_Year','Production']]
    print(production_year_wise)

    ######################################################
    # if the slope is a +ve value --> increasing trend
    # if the slope is a -ve value --> decreasing trend
    # if the slope is a zero value --> No trend

    def trendline(index,data, order=1):
      coeffs = np.polyfit(index, list(data), order)
      slope = coeffs[-2]
      return float(slope)

    index=production_year_wise['Crop_Year']
    List=production_year_wise['Production']
    resultent=trendline(index,List)
    print('\nslope:',resultent)
    print('\n')
    if(resultent<=0):
      print(crop,'is not profitable crop in your area and has a declined production rate')
      print("####################################\n")
    elif (resultent>0 and resultent<=500) :
      print(crop,'is not that profitable but OK crop to be grown in your area since was at peak at some years')
      print("####################################\n")
    else:
      print(crop,'is profitable crop in your area and has a continuous inclined production rate')
      print("####################################\n")
    #########################################################
    production_year_wise.plot(kind='line',x='Crop_Year',y='Production')
    plt.title(crop)
    plt.show()
  else:
    print(crop,'not available in existing dataset for your district\n ')
    print("####################################\n")
  il+=1
##################################################################################################  
#================================================================================================
#pesticides
print('PESTICIDES')
pesticides=pd.read_csv('pesticides.csv')
print('================================================================')
il=0
while il<len(newlist):
  crop=newlist[il]
  print('Diseases and pesticides for',crop)
  print('\n')
  crop=crop.upper()
  data31=pesticides[pesticides.Crop.str.contains(crop)] 
  isempty = data31.empty
  if isempty==False:
    pesticidesdata=data31[['Diseases','Pesticide']]
    print(pesticidesdata)
    print('\n\n')

  else:
    print(crop,'has no disease in dataset\n ')
    print("####################################\n")
  il+=1

##########################################################################################################  
##########################################################################################################
#Loans
print('\nLOAN SECTION')
#tp=firebase.get('/Production Cost',None)

fund=200000
area=3
l3=[]
l3.append(fund)
l3.append(area)
list3=[l3]
Available_Fund=list3[0][0]
Available_Area=list3[0][1]

costdata=pd.read_csv('cost_of_production.csv') 
il=0

loanstr=[]
while il<len(newlist):
  crop=newlist[il]
  crop=crop.capitalize()
  #get the cost per hectar for predicted crop
  costdata1=np.array(costdata[costdata.Crop.str.contains(crop)])
  isempty=False
  if len(costdata1)==0:
    isempty=True
  if isempty==False:
    print('===================================================================================================')
    print(crop.upper())
    loanstr.append(crop.upper())
    print('\n')
    print('Average price of steps required for',crop,'production:')
    loanstr.append('Average price of steps required for '+crop+' production:')
    print('Field Preparation:',costdata1[0][1])
    loanstr.append('Field Preparation: '+str(costdata1[0][1]))
    print('Nursery and planting sowing:',costdata1[0][2])
    loanstr.append('Nursery and planting sowing: '+str(costdata1[0][2]))
    print('Weeding:',costdata1[0][3])
    loanstr.append('Weeding: '+str(costdata1[0][3]))
    print('Plant protection:',costdata1[0][4])
    loanstr.append('Plant protection: '+str(costdata1[0][4]))
    print('Fertilizers:',costdata1[0][5])
    loanstr.append('Fertilizers: '+str(costdata1[0][5]))
    print('Wages:',costdata1[0][6])
    loanstr.append('Wages:' +str(costdata1[0][6]))
    print('Staking, transport and other expenses:',costdata1[0][7])
    loanstr.append('Staking, transport and other expenses: '+str(costdata1[0][7]))
    print('\n')

    total=costdata1[0][8]
    print('Total cost required to cultivate 1 hectare:',total)
    loanstr.append('Total cost required to cultivate 1 hectare: '+str(total))
    Funds_Required = Available_Area*total
    print('Required Fund for successfull cultivation of',Available_Area,'hectares:',Funds_Required)
    loanstr.append('Required Fund for successfull cultivation of '+str(Available_Area)+' hectares: '+str(Funds_Required))
    print('Available Funds:',Available_Fund)
    loanstr.append('Available Funds: '+str(Available_Fund))
    
    if(Funds_Required>Available_Fund):
      loan=Funds_Required-Available_Fund
      print('\nYou need to apply for a loan of:',loan)
      loanstr.append('You need to apply for a loan of: '+str(loan))
      print('\n\n')
    else:
      print('\nYour Funds are suffucient for cultivation')
      loanstr.append('Your Funds are suffucient for cultivation')
      print('\n\n')
  else:
    print('No Data Found for',crop,'to manage cost')
    print('\n\n')
  print('===================================================================================================') 
  loanstr.append('##########################################################################################')
  il+=1
#################################################################################################################

#Bank Loan
print('Bank Available with loan:\n')
costdata=pd.read_csv('loans.csv')
print(costdata)
print('\n')

####################################################################################################

##########################################################################################################
#SMS
# import requests
# import json

# reqUrl = 'https://www.sms4india.com/api/v1/sendCampaign'

# # get request
# def sendPostRequest(reqUrl, apiKey, secretKey, useType, phoneNo, senderId, textMessage):
#   req_params = {
#   'apikey':apiKey,
#   'secret':secretKey,
#   'usetype':useType,
#   'phone': phoneNo,
#   'message':textMessage,
#   'senderid':senderId
#   }
#   return requests.post(reqUrl, req_params)
# #pcrops=crops[pos[0]]+','+crops[pos[1]]+','+crops[pos[2]]+','+crops[pos[3]]+','+crops[pos[4]]

# # get response
# response = sendPostRequest(reqUrl, '4C7UVPYA9NZHQJNSRXLOL4MSAJLY43Q4', 'ZVAI1EYFB0IFDXAJ', 'stage', '7028652300', 'yashkurekar@gmail.com', sms_crops )
# response = sendPostRequest(reqUrl, '4C7UVPYA9NZHQJNSRXLOL4MSAJLY43Q4', 'ZVAI1EYFB0IFDXAJ', 'stage', '7410754757', 'yashkurekar@gmail.com', sms_crops )
# # print response if you want
# #print(response.text)

#############################################################################################################