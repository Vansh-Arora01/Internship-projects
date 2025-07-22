    # PYTHON 1 PROJECT 
    # MULTIPLE DISEASE PREDICTOR 

#  (IMPORT THE NECESSARY LIBRARY)
import pandas as pd

df = pd.read_csv('https://github.com/YBI-Foundation/Dataset/raw/refs/heads/main/MultipleDiseasePrediction.csv')

df.head()
df.info(verbose =True)
df.columns

# // (Defining the output)
y= df['prognosis']

# // (Defining the inputs)

x= df.drop(['prognosis'], axis=1)

#(Splitting)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.2, random_state = 2529)

# (Model Sellection)

from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier()

model.fit(x_train, y_train)

# (3 predict test)

y_predict = model.predict(x_test)
x_test.iloc[0]

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
print (classification_report(y_test, y_predict))