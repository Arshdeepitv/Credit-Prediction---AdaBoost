from flask import Flask, render_template, request
import pickle
import model as m
import pandas as pd

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/classify')
def classify():
    return render_template('classify.html')

@app.route('/predict', methods=['POST', 'GET'])
def predict():
    x = dict(request.form)
    df = pd.DataFrame(x, index=[0])
    checking_status_dictionary = {'less than 0':'<0', "between 0 and 200":'0<=X<200', "no checking":'no checking', "greater than 200":'>=200'}
    savings_status_dictionary = {"no known savings":'no known savings', "less than 100":'<100', "between 100 and 500":'100<=X<500', "between 500 and 1000":'500<=X<1000', "greater than 1000":'>=1000'}
    employment_dictionary = {"unemployed":'unemployed', "less than 1 year":'<1', "1 to 4 years":'1<=X<4', "4 to 7 years":'4<=X<7', "more than 7 years":'>=7'}
    df['checking_status'] = checking_status_dictionary[df.loc[0, 'checking_status']]
    df['savings_status'] = savings_status_dictionary[df.loc[0, 'savings_status']]
    df['employment'] = employment_dictionary[df.loc[0, 'employment']]
    for col in df.columns:
        try:    
            df[col] = df[col].astype('float64')
        except:
            pass
    df = m.process(df)
    model = pickle.load(open('model.sav', 'rb'))
    pred = model.predict(df)
    pred = m.enc.inverse_transform(pred)
    pred = pred[0]
    return render_template('predict.html', pred=pred)

if __name__=='__main__':
    app.run(debug=True)