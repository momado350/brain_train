import numpy as np
import pandas as pd


brains = pd.read_csv('./brain.csv')
x = brains["weight"].values.reshape(-1, 1)
y = brains["size"].values.reshape(-1, 1)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(x, y, random_state=42)



from sklearn.linear_model import LinearRegression

model = LinearRegression()

model.fit(X_train, y_train)

from sklearn.metrics import mean_squared_error, r2_score

predicted = model.predict(X_test)
print(predicted)

#import pickle
#with open('/Users/momad/Desktop/brain/rf.pkl', 'wb') as model_pkl:
#   pickle.dump(model, model_pkl, protocol=2)
    
from joblib import dump

dump(model, '/Users/momad/Desktop/brain/br.joblib')