import pandas as pd
data=pd.read_csv('WineQT.csv')
print(data.columns)
drop=data['alcohol'].drop
data['fixed acidity']=data['fixed acidity'].fillna(data['fixed acidity'].mean())
data['volatile acidity']=data['volatile acidity'].fillna(data['volatile acidity'].mean())
data['citric acid']=data['citric acid'].fillna(data['volatile acidity'].mean())
data['residual sugar']=data['residual sugar'].fillna(data['residual sugar'].mean())
data['alcohol']=data['alcohol'].fillna(data['alcohol'].mean())
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(data[['fixed acidity','volatile acidity','citric acid','residual sugar','chlorides','free sulfur dioxide','total sulfur dioxide','density','pH','sulphates']],data['alcohol'],test_size=0.35)
import keras.activations
from keras.models import Sequential
from keras.layers import Dense
import keras.losses
import keras.metrics
model=Sequential()
model.add(Dense(input_dim=data[['fixed acidity','volatile acidity','citric acid','residual sugar','chlorides','free sulfur dioxide','total sulfur dioxide','density','pH','sulphates']].shape[1],units=2,activation=keras.activations.tanh))
model.add(Dense(input_dim=data[['fixed acidity','volatile acidity','citric acid','residual sugar','chlorides','free sulfur dioxide','total sulfur dioxide','density','pH','sulphates']].shape[1],units=2,activation=keras.activations.tanh))
model.add(Dense(input_dim=data[['fixed acidity','volatile acidity','citric acid','residual sugar','chlorides','free sulfur dioxide','total sulfur dioxide','density','pH','sulphates']].shape[1],units=2,activation=keras.activations.linear))
model.add(Dense(units=1,activation='linear'))
model.compile(optimizer='adam',loss=keras.losses.mean_absolute_error,metrics='mae')
model.fit(x_train,y_train,epochs=13,batch_size=10)
pred=model.predict([[5.6,0.615,0.0,1.6,0.08900000000000001,16.0,59.0,0.9943,3.58,0.52]])
print(pred)