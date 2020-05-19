##############################################################
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
import numpy as np
import matplotlib.pyplot as plt
from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd
import shutil
import os



app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))
stg=[]
stat=[]
pos=[-1]*5
sms_crops='Predicted crops are:'
crops=['APPLE','BANANA','BLACK GRAM','CHICKPEA','COCONUT','COFFEE','COTTON','GRAPES','GROUNDNUT','JUTE','KIDNEY BEANS','LENTIL','MAIZE','MANGO','MILLET','MOTH BEANS','MUNG BEAN','MUSK MELON','ORANGE','PAPAYA','PEAS','PIGEON PEAS','POMEGRANATE','RICE','RUBBER','SUGARCANE','TEA','TOBACCO','WATERMELON','WHEAT']
cr='rice'

str_crop=[]
str_no_crop=[]
sg2=[]
loanstr=[]
costdata=pd.read_csv('loans.csv')
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():

    shutil.rmtree('static/images')
    os.makedirs('static/images')
    int_features = [float(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    l=[]
    for i in range(0,4):
        l.append(final_features[0][i])
        
    predictcrop=[l]
    
    data = pd.read_csv('cpdata.csv')
    
    label= pd.get_dummies(data.label).iloc[: , 1:]
    data= pd.concat([data,label],axis=1)
    data.drop('label', axis=1,inplace=True)
    train=data.iloc[:, 0:4].values
    test=data.iloc[: ,4:].values  



    
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(train, test, test_size = 0.2, random_state = 0)
    
    
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)


    fi=sc.transform(np.array(predictcrop))
    
    predictions = model.predict(fi)




    import copy
    prediction1 = copy.copy(predictions)
    prediction1.sort()
    max=[prediction1[0][29],prediction1[0][28],prediction1[0][27],prediction1[0][26],prediction1[0][25]]
    # pos=[-1]*5
    for k in range (0,5):
        for j in range(0,30):
            if(max[k]==predictions[0][j] and max[k]>=0.100000):
                pos[k]=j

   
    output=[]
  
    for i in range(0,5):
        if(pos[i]!=-1):
            output.append(crops[pos[i]])
    
    if i==0:
        if(pos[i]!=-1):
            temp=crops[pos[i]]
            sms_crops+=temp.capitalize()+','
    if i==1:
        if(pos[i]!=-1):
            temp=crops[pos[i]]
            sms_crops+=temp.capitalize()+','
    if i==2:
        if(pos[i]!=-1):
            temp=crops[pos[i]]
            sms_crops+=temp.capitalize()+','
    if i==3:
        if(pos[i]!=-1):
            temp=crops[pos[i]]
            sms_crops+=temp.capitalize()+','
    if i==4:
        if(pos[i]!=-1):
            temp=crops[pos[i]]
            sms_crops+=temp.capitalize()+','
 
    
        

    tenure=pd.read_csv('tenure.csv')
    tenure_you_want=12 
    il=0

    while il<5:
        if(pos[il]!=-1):
            crop=crops[pos[il]]
            str_crop.append(crop)
            tenuredata=tenure[tenure.Crops.str.contains(crop)]
            isempty=tenuredata.empty
            temp='Time period required for cultivation is '+str(float(tenuredata['min.time']))+' to '+str(float(tenuredata['max.time']))+' months'
            str_crop.append(temp)
            temp='provided tenure: '+str(tenure_you_want)+' months'
            str_crop.append(temp)
      
            if isempty==False:
                if ((float(tenuredata['min.time'])<=tenure_you_want) and (tenure_you_want<=float(tenuredata['max.time']) or tenure_you_want>=float(tenuredata['max.time']))):
                    temp=crop+' can be grown in provided tenure'
                    str_crop.append(temp)
                else:
                    temp=crop+' cannot be grown in period you want'
                    str_crop.append(temp)
                    pos[il]=-1
  
        il+=1  
    
    str_crop.append('Crops that match to the provided tenure are:')
    il=0
    count=0
    newlist=[]
    while il<5:
        if pos[il]!=-1:
            crop=crops[pos[il]]
            print(crop)
            str_crop.append(crop)
            newlist.append(crop)
            count+=1
        il+=1  
    
    if(count==0):
        str_no_crop.append('No crop can be cultivated in tenure given by you')



    loc=[]
    loc.append('Maharashtra')
    loc.append('Chandrapur')  
    location=[loc]
    
    district_name=location[0][1].upper()
    State_Name=location[0][0]
    
    data2=pd.read_csv('apy.csv')  
    il=0

    while il<len(newlist):
        crop=newlist[il]
        crop=crop.capitalize()
        data3=data2[data2.Crop.str.contains(crop) & data2.District_Name.str.contains(district_name)&data2.State_Name.str.contains(State_Name)] 
        isempty = data3.empty
        if isempty==False:
            production_year_wise=data3[['Crop_Year','Production']]
            def trendline(index,data, order=1):
                coeffs = np.polyfit(index, list(data), order)
                slope = coeffs[-2]
                return float(slope)

            index=production_year_wise['Crop_Year']
            List=production_year_wise['Production']
            resultent=trendline(index,List)

            if(resultent<=0):
                stat.append(crop+' is not profitable crop in your area and has a declined production rate')
                stat.append("####################################")
            elif (resultent>0 and resultent<=500) :
                stat.append(crop+' is not that profitable but OK crop to be grown in your area since was at peak at some years')
                stat.append("####################################")
            else:
                stat.append(crop+' is profitable crop in your area and has a continuous inclined production rate')
                stat.append("####################################")
            production_year_wise.plot(kind='line',x='Crop_Year',y='Production')
            plt.title(crop)
            plt.savefig('static/images/crop'+str(il)+'.png')          
        else:
            stat.append(crop+' not available in existing dataset for your district')
            stat.append("####################################")
        il+=1

    

    pesticides=pd.read_csv('pesticides.csv')
    il=0
    while il<len(newlist):
        crop=newlist[il]
        crop=crop.upper()
        data31=pesticides[pesticides.Crop.str.contains(crop)] 
        isempty = data31.empty
        if isempty==False:
            pesticidesdata=data31[['Diseases','Pesticide']]
            sg2.append(pesticidesdata)


        else:
            str_no_crop.append((crop+' has no disease in dataset'))

        il+=1



    
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
    

    while il<len(newlist):
        crop=newlist[il]
        crop=crop.capitalize()
        costdata1=np.array(costdata[costdata.Crop.str.contains(crop)])
        isempty=False
        if len(costdata1)==0:
            isempty=True
        if isempty==False:
            loanstr.append(crop.upper())
            loanstr.append('Average price of steps required for '+crop+' production:')
            loanstr.append('Field Preparation: '+str(costdata1[0][1]))
            loanstr.append('Nursery and planting sowing: '+str(costdata1[0][2]))
            loanstr.append('Weeding: '+str(costdata1[0][3]))
            loanstr.append('Plant protection: '+str(costdata1[0][4]))
            loanstr.append('Fertilizers: '+str(costdata1[0][5]))
            loanstr.append('Wages:' +str(costdata1[0][6]))
            loanstr.append('Staking, transport and other expenses: '+str(costdata1[0][7]))

            total=costdata1[0][8]
            loanstr.append('Total cost required to cultivate 1 hectare: '+str(total))
            Funds_Required = Available_Area*total
            loanstr.append('Required Fund for successfull cultivation of '+str(Available_Area)+' hectares: '+str(Funds_Required))
            loanstr.append('Available Funds: '+str(Available_Fund))
    
            if(Funds_Required>Available_Fund):
                loan=Funds_Required-Available_Fund
                loanstr.append('You need to apply for a loan of: '+str(loan))
            else:
                loanstr.append('Your Funds are suffucient for cultivation')
        

        loanstr.append('##########################################################################################')
        il+=1


    return render_template('index.html', len=len(output),output=output)



@app.route('/tenure',methods=['POST'])
def tenure():    

    return render_template('tenure.html', output=str_crop, str_no_crop = str_no_crop)

@app.route('/statistics',methods=['POST'])
def statistics():    

    return render_template('statistics.html', output=stat)

@app.route('/pesticides',methods=['POST'])
def pesticides():  
    
    if(len(sg2)==0):
        return render_template('pesticides.html', sg2='No data found') 
    elif(len(sg2)==1):
        return render_template('pesticides.html', table0 = sg2[0].to_html (header = 'true')) 
    elif(len(sg2)==2):
        return render_template('pesticides.html', table0 = sg2[0].to_html (header = 'true'),table1 = sg2[1].to_html (header = 'true'))
    elif(len(sg2)==3):
        return render_template('pesticides.html', table0 = sg2[0].to_html (header = 'true'),table1 = sg2[1].to_html (header = 'true'),table2 = sg2[2].to_html (header = 'true'))
    elif(len(sg2)==4):
        return render_template('pesticides.html', table0 = sg2[0].to_html (header = 'true'),table1 = sg2[1].to_html (header = 'true'),table2 = sg2[2].to_html (header = 'true'),table3 = sg2[3].to_html (header = 'true'))
    elif(len(sg2)==5):
        return render_template('pesticides.html', table0 = sg2[0].to_html (header = 'true'),table1 = sg2[1].to_html (header = 'true'),table2 = sg2[2].to_html (header = 'true'),table3 = sg2[3].to_html (header = 'true'),table4 = sg2[4].to_html (header = 'true'))


@app.route('/managecost',methods=['POST'])
def managecost():    

    return render_template('managecost.html', output=loanstr)

@app.route('/banks',methods=['POST'])
def banks():    

    return render_template('/banks.html', table0=costdata.to_html (header = 'true'))

if __name__ == "__main__":
    app.run(debug=True)