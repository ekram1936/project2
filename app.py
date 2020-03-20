from flask import Flask,render_template,request,redirect,url_for,flash
from flask_bootstrap import Bootstrap 
from flask_uploads import UploadSet,configure_uploads,IMAGES,DATA,ALL
from flask_sqlalchemy import SQLAlchemy 
from flask_material import Material
from sklearn.externals import joblib
from werkzeug import secure_filename


# EDA Packages
import pandas as pd 
import numpy as np 

# ML Packages
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from werkzeug import secure_filename
import os
import datetime
import time

import mysql.connector
app = Flask(__name__)
Bootstrap(app)
Material(app)


files = UploadSet('files',ALL)
app.config['UPLOADED_FILES_DEST'] = 'static/csv_files'
configure_uploads(app,files)







con = mysql.connector.connect(host= "localhost" , user= "root" , passwd="ekramhossain", database = "project_3")
cursor=con.cursor() 
@app.route('/login')
def login():
	return render_template("login.html")


@app.route('/register')
def register():
	return render_template("register.html")

@app.route('/home')
def home():
	return render_template("home.html")

@app.route('/validation', methods=['GET','POST'])
def validation():
	username =request.form.get('username')
	password =request.form.get('password')
	cursor.execute('SELECT * FROM user WHERE username = %s AND pass = %s', (username, password))
	users=cursor.fetchall()
	if len(users)>0:
		return redirect('/home')
	else:
		return redirect('/login')

@app.route("/add",methods=["POST"])
def signUp():
	username = str(request.form["username"])
	password = str(request.form["password"])

	email = str(request.form["email"])
	
	cursor = con.cursor()
	
	cursor.execute("INSERT INTO user (username,email,pass)VALUES(%s,%s,%s)",(username,email,password))
	con.commit()
	return redirect(url_for("login"))





@app.route('/dataupload',methods=["GET","POST"])
def dataupload():
	if request.method == 'POST' and 'csv_data' in request.files:
		file = request.files['csv_data']
		filename = secure_filename(file.filename)
		file.save(os.path.join('static/csv_files',filename))
		fullfile = os.path.join('static/csv_files',filename)
		date = str(datetime.datetime.fromtimestamp(time.time()).strftime("%Y-%m-%d %H:%M:%S"))

		# EDA function
		df = pd.read_csv(os.path.join('static/csv_files',filename))
		df_size = df.size
		df_shape = df.shape
		df_columns = list(df.columns)
		df_targetname = df[df.columns[-1]].name
		df_featurenames = df_columns[0:-1] # select all columns till last column
		df_Xfeatures = df.iloc[:,0:-1] 
		df_Ylabels = df[df.columns[-1]] # Select the last column as target
		# same as above df_Ylabels = df.iloc[:,-1]
		from sklearn.preprocessing import LabelEncoder, OneHotEncoder
		labelencoder_y = LabelEncoder()
		df_Ylabels = labelencoder_y.fit_transform(df[df.columns[-1]])
		

		# Model Building
		X = df_Xfeatures
		Y = df_Ylabels
		seed = 7
		# prepare models
		models = []
		models.append(('LR', LogisticRegression()))
		models.append(('LDA', LinearDiscriminantAnalysis()))
		models.append(('KNN', KNeighborsClassifier()))
		models.append(('CART', DecisionTreeClassifier()))
		models.append(('NB', GaussianNB()))
		models.append(('SVM', SVC()))
		# evaluate each model in turn
		

		results = []
		names = []
		allmodels = []
		scoring = 'accuracy'
		for name, model in models:
			kfold = model_selection.KFold(n_splits=10, random_state=seed)
			cv_results = model_selection.cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
			results.append(cv_results)
			names.append(name)
			msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
			allmodels.append(msg)
			model_results = results
			model_names = names		
		
	return render_template('details.html',filename=filename,date=date,df_size=df_size,df_shape=df_shape,df_columns =df_columns,df_targetname =df_targetname,model_results = allmodels,model_names = names,fullfile = fullfile,dfplot = df)

@app.route('/preview')
def preview():
    df = pd.read_csv("data/iris.csv")
    return render_template("preview.html",df_view = df)

@app.route('/view')
def view():
    return render_template("view.html")

@app.route('/analyze',methods=["GET","POST"])
def analyze():
	if request.method == 'POST':
		petal_length = request.form['petal_length']
		sepal_length = request.form['sepal_length']
		petal_width = request.form['petal_width']
		sepal_width = request.form['sepal_width']
		model_choice = request.form['model_choice']

		# Clean the data by convert from unicode to float 
		sample_data = [sepal_length,sepal_width,petal_length,petal_width]
		clean_data = [float(i) for i in sample_data]

		# Reshape the Data as a Sample not Individual Features
		ex1 = np.array(clean_data).reshape(1,-1)

		# ex1 = np.array([6.2,3.4,5.4,2.3]).reshape(1,-1)

		# Reloading the Model
		if model_choice == 'logitmodel':
		    logit_model = joblib.load('data/logit_model_iris.pkl')
		    result_prediction = logit_model.predict(ex1)
		elif model_choice == 'knnmodel':
			knn_model = joblib.load('data/knn_model_iris.pkl')
			result_prediction = knn_model.predict(ex1)
		elif model_choice == 'svmmodel':
			knn_model = joblib.load('data/svm_model_iris.pkl')
			result_prediction = knn_model.predict(ex1)

	return render_template('view.html', petal_width=petal_width,
		sepal_width=sepal_width,
		sepal_length=sepal_length,
		petal_length=petal_length,
		clean_data=clean_data,
		result_prediction=result_prediction,
		model_selected=model_choice)



app.secret_key = 'many random bytes'


@app.route('/admin')
def admin():
    con = mysql.connector.connect(host= "localhost" , user= "root" , passwd="ekramhossain", database = "project_3")
    cur=con.cursor() 
    cur.execute("SELECT * FROM user")
    data = cur.fetchall()
    cur.close()




    return render_template('index2.html', user=data )


@app.route('/datasetUpload')
def datasetUpload():
    con = mysql.connector.connect(host= "localhost" , user= "root" , passwd="ekramhossain", database = "project_3")
    cur=con.cursor() 
    cur.execute("SELECT * FROM dataset")
    data = cur.fetchall()
    cur.close()




    return render_template('datasetUpload.html', dataset=data )


@app.route('/insert', methods = ['POST','GET'])
def insert():

    if request.method == "POST":
        flash("Data Inserted Successfully")
        name = request.form['name']
        file = request.files['file']
        descrip = request.form['description']
        con = mysql.connector.connect(host= "localhost" , user= "root" , passwd="ekramhossain", database = "project_3")
        cur=con.cursor() 
        cur.execute("INSERT INTO dataset (name, file, descrip) VALUES (%s, %s, %s)", (name, file, descrip))
       
        return redirect(url_for('datasetUpload'))




@app.route('/delete/<string:id_data>', methods = ['GET'])
def delete(id_data):
    flash("Record Has Been Deleted Successfully")
    con = mysql.connector.connect(host= "localhost" , user= "root" , passwd="ekramhossain", database = "project_3")
    cur=con.cursor() 
    cur.execute("DELETE FROM user WHERE username=%s", (id_data,))
    
    return redirect(url_for('Index'))

@app.route('/deletedata/<string:id_data>', methods = ['GET'])
def deletedata(id_data):
    flash("Record Has Been Deleted Successfully")
    con = mysql.connector.connect(host= "localhost" , user= "root" , passwd="ekramhossain", database = "project_3")
    cur=con.cursor() 
    cur.execute("DELETE FROM dataset WHERE name=%s", (id_data,))
    
    return redirect(url_for('datasetUpload'))





@app.route('/update',methods=['POST','GET'])
def update():

    if request.method == 'POST':
        id_data = request.form['id']
        name = request.form['name']
        email = request.form['email']
        phone = request.form['phone']
        con = mysql.connector.connect(host= "localhost" , user= "root" , passwd="ekramhossain", database = "project_3")
        cur = mysql.connector.cursor()
        cur.execute("""
               UPDATE students
               SET name=%s, email=%s, phone=%s
               WHERE id=%s
            """, (name, email, phone, id_data))
        flash("Data Updated Successfully")
        return redirect(url_for('Index'))





if __name__ == '__main__':
	app.run(debug=True)