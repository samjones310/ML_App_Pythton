import pandas
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
import pickle
#url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
dataframe = pandas.read_csv("heart.csv", header =[1])
array = dataframe.values
X = array[:,0:13]
Y = array[:,13]
test_size = 0.33
seed = 7
X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, test_size=test_size, random_state=seed)
# Fit the model on 33%
model = LogisticRegression()
model.fit(X_train, Y_train)
# save the model to disk
filename = 'model.pkl'
pickle.dump(model, open(filename, 'wb'))
 
# some time later...
 
# load the model from disk
loaded_model = pickle.load(open(filename, 'rb'))
print(X_test)
print(model)
de=[[63,1,3,145,233,1,0,150,0,2.3,0,0,1]]
pre= loaded_model.predict(de)
print(pre)
result = loaded_model.score(X_test, Y_test)
print(result)
