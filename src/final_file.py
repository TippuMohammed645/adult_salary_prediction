import pandas as pd
import joblib
from sklearn.preprocessing import OneHotEncoder
from flask import Flask,request,jsonify,render_template
import numpy as np
import logging

#configuring logging
#The format of the logging messages includes the timestamp, logging level, and message.
logging.basicConfig(level=logging.info,format='%(asctime)s-%(levelname)s-%(message)s')

# load the model ad cleaning methods 
logging.info("loading models and transformers")
one_hencoder=joblib.load('1hot_model')
scaler_model=joblib.load('scale_model')

with open('final_model1','rb') as model_file:
    model=joblib.load(model_file)

app=Flask(__name__)
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/Predict',methods=['POST'])
def Predict(): 
    #get the data from the post request method
    #request.form.to_dict() converts all form data into a Python dictionary where keys are field names and values are the corresponding field values.
    #data=request.form.to_dict()
    try:
        logging.info("recieved prediction request")
        input_dict={
                    'age': int(request.form['Age']),
                    'workclass': request.form['Workclass'],
                    'fnlwgt': int(request.form['Fnlwgt']),
                    'education-num': int(request.form['Education-num']),
                    'occupation': request.form['Occupation'],
                    'relationship': request.form['Relationship'],
                    'marital-status':request.form['Marital-status'],
                    'race': request.form['Race'],
                    'sex': request.form['Sex'],
                    'capital-gain': int(request.form['Capital-gain']),
                    'capital-loss': int(request.form['Capital-loss']),
                    'hours-per-week': int(request.form['Hours-per-week']),
                    'country': request.form['Native-country']
                }
        logging.info("converting ip data to dataframe")
        #convert the data into a dataframe
        input_data=pd.DataFrame(input_dict,index=[0])
        
        #fetching category and numeric variables
        categ_features=input_data.select_dtypes(include='object').columns
        num_features=input_data.select_dtypes(exclude='object').columns
        
        logging.info("Cleaning categorical features.")
        # clean the input data using preloaded transformers
        clean_data1=one_hencoder.transform(input_data[categ_features]).toarray()
        onc=one_hencoder.named_transformers_['cat_encoding'].named_steps['onehot']
        feature_names=onc.get_feature_names_out()
        clean_data1_=pd.DataFrame(clean_data1,columns=feature_names)
        logging.debug(f"categorical features cleaned :{clean_data1_.head()}")
        
        logging.info("Cleaning numerical features.")
        clean_data2=scaler_model.transform(input_data[num_features])
        clean_data2_=pd.DataFrame(clean_data2,columns=num_features)
        logging.debug(f"Numerical features cleaned: {clean_data2_.head()}")
        
        #concate the cleaned data 
        clean_data=pd.concat([clean_data1_,clean_data2_],axis=1)
        logging.debug(f"Final cleaned data: {clean_data.head()}")
        
        #make prediction on the cleaned data
        logging.info("Making prediction.")
        predicted_value=model.predict(clean_data)
        logging.debug(f"Predicted value: {predicted_value}")
        
        if predicted_value==1:
            prediction= 'The salary would probabbly be  >50k'
        else:
            prediction= 'The salary would probabbly be =<50k'
        
        return render_template('result.html',prediction=prediction)
    
    except Exception as e:
        logging.error(f"error during prediction :{e}")
        return "Error during prediction",500

if __name__=='__main__':
    app.run(debug=True,port=5001)