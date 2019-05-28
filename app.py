from flask import Flask,render_template,url_for,request
from flask_bootstrap import Bootstrap 
import pandas as pd 
import numpy as np 

# ML Packages
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.externals import joblib


app = Flask(__name__)
Bootstrap(app)


@app.route('/')
def index():
	return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
	'''df= pd.read_csv("heart.csv")
	df_X = 
	df_Y = df.target
    
    # Vectorization
	corpus = df_X
	cv = CountVectorizer()
	X = cv.fit_transform(corpus) '''
	
	# Loading our ML Model
	linear_model = open("models/model.pkl","rb")
	clf = joblib.load(linear_model)

	# Receives the input query from form
	if request.method == 'POST':
		age = request.form['age']
		sex = request.form['sex']
		cp = request.form['cp']
		trestbps = request.form['trestbps']
		chol = request.form['chol']
		fps = request.form['fps']
		restecg = request.form['restecg']
		thalach = request.form['thalach']
		exang = request.form['exang']
		oldpeak = request.form['oldpeak']
		slope = request.form['slope']
		ca = request.form['ca']
		thal = request.form['thal']
		data = [[int(age),int(sex),int(cp),int(trestbps),int(chol),int(fps),int(restecg),int(thalach),int(exang),int(oldpeak),int(slope),int(ca),int(thal)]]
		print(data)
		#vect = cv.transform(data).toarray()
		my_prediction = clf.predict(data)
		print(my_prediction)
	return render_template('results.html',prediction = my_prediction,name ="sam jones")


if __name__ == '__main__':
	app.run(debug=True)
