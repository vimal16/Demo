from flask import Flask,render_template,request
import pandas as pd
import numpy as np

app = Flask(__name__)
 
@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/data', methods=['GET','POST'])
def data():
    if request.method=='POST':
        file=request.form['upload-file']
        data = pd.read_csv(file)
        data['year'] = pd.DatetimeIndex(data['TIME']).year
        data['month'] = pd.DatetimeIndex(data['TIME']).month
        data['day'] = pd.DatetimeIndex(data['TIME']).day
        data['hour'] = pd.DatetimeIndex(data['TIME']).hour
        data['min'] = pd.DatetimeIndex(data['TIME']).minute
        
        from sklearn import ensemble
        X_train,y_train=data.drop("ACTUAL",axis=1),data["ACTUAL"]
        cols = X_train.columns
        X_train[cols] = X_train[cols].apply(pd.to_numeric, errors='coerce').fillna(0) 
        est = ensemble.RandomForestRegressor().fit(X_train,y_train)
        data['FORECAST'] = est.predict(np.array(X_train))
        data = data[["TIME","ACTUAL","FORECAST"]]
        return render_template('data.html',data=data.to_html(index = False))
            
        



if __name__=="__main__":
    app.run()
