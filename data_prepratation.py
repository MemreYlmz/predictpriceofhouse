#import pandas as pd
#import sqlite3
#import joblib
#from sklearn.model_selection import train_test_split
#from sklearn.pipeline import Pipeline
#from sklearn.preprocessing import StandardScaler, OneHotEncoder
#from sklearn.compose import ColumnTransformer
#from sklearn.ensemble import RandomForestClassifier
#
#
#
## Veriyi yükleme ve temizleme
#df = pd.read_csv("data_cleaned2.csv")
#df["city"] = df["city"].astype("category")
#df["district"] = df["district"].astype("category")
#df["neighborhood"] = df["neighborhood"].astype("category")
#df["room"] = df["room"].astype("int")
#df["living_room"] = df["living_room"].astype("int")
#df["area"] = df["area"].astype("int")
#df["age"] = df["age"].astype("int")
#df["floor"] = df["floor"].astype("int")
#df["price"] = df["price"].astype("int")
#
## Pipeline oluşturma
#categorical_features = ["city", "district", "neighborhood"]
#numerical_features = ["room", "living_room", "area", "age", "floor"]
#full_pipeline = ColumnTransformer([
#    ("num", StandardScaler(), numerical_features),
#    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
#])
#
## Model eğitme
#X = df.drop("price", axis=1)
#y = df["price"]
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#model = Pipeline([
#    ("preparation", full_pipeline),
#    ("model", RandomForestClassifier(n_estimators=100))
#])
#model.fit(X_train, y_train)
#
#
#new_data = pd.DataFrame({
#    "city":["İzmir"],
#    "district":["bornova"],
#    "neighborhood":["erzene"],
#    "room":[5],
#    "living_room":[1],
#    "area":[80],
#    "age":[5],
#    "floor":[3],
#})
#print(model.predict(new_data))
#print(df[(df["city"]=="izmir")&(df["district"]=="bornova")&(df["neighborhood"]=="erzene")])
#
## Model ve pipeline'ı kaydetme
#joblib.dump(model, 'model.pkl')
#joblib.dump(full_pipeline, 'pipeline.pkl')
#
## Veritabanına kaydetme
#conn = sqlite3.connect('hepsiemlak.db')
#df.to_sql('emlak_verileri', conn, if_exists='replace', index=False)
#conn.close()


#DENEME

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import sqlite3
import joblib
df = pd.read_csv("data_cleaned2.csv")

df["city"] = df["city"].astype("category")
df["district"] = df["district"].astype("category")
df["neighborhood"] = df["neighborhood"].astype("category")
df["room"] = df["room"].astype("int")
df["living_room"] = df["living_room"].astype("int")
df["area"] = df["area"].astype("int")
df["age"] = df["age"].astype("int")
df["floor"] = df["floor"].astype("int")
df["price"] = df["price"].astype("int")

categorical_features = ["city","district","neighborhood"]
numerical_features = ["room","living_room","area","age","floor"]

full_pipeline = ColumnTransformer([
    ("num",StandardScaler(),numerical_features),
    ("cat",OneHotEncoder(handle_unknown="ignore"),categorical_features)
])

X = df.drop("price",axis=1)
y = df["price"]

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

model = Pipeline([
    ("preparation",full_pipeline),
    ("model",LinearRegression())
])

model.fit(X_train,y_train)

y_pred = model.predict(X_test)
mse = mean_squared_error(y_test ,y_pred)
rmse = np.sqrt(mse)
r2=r2_score(y_test,y_pred)

new_data = pd.DataFrame({
    "city":["İzmir"],
    "district":["karabağlar"],
    "neighborhood":["refet_bele"],
    "room":[3],
    "living_room":[1],
    "area":[120],
    "age":[0],
    "floor":[3],
})
print(model.predict(new_data))


## Model ve pipeline'ı kaydetme
joblib.dump(model, 'model.pkl')
joblib.dump(full_pipeline, 'pipeline.pkl')

# Veritabanına kaydetme
conn = sqlite3.connect('hepsiemlak.db')
df.to_sql('emlak_verileri', conn, if_exists='replace', index=False)
conn.close()




