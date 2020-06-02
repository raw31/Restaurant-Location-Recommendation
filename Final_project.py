import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


dataset = pd.read_csv("zomato.csv")

# Removing / from rates column 
dataset['rate'] = dataset['rate'].apply(lambda x: str(x).split('/')[0])

# Handling string values and converting to integer eg: 1,200 to 1200

dataset['approx_cost(for two people)'] = dataset['approx_cost(for two people)'].str.replace(',', '').astype(float)

# deleting unnecessary features

del dataset['url']
del dataset['address']
del dataset['phone']
del dataset['reviews_list']
del dataset['menu_item']

# Handling NEW restaurents from the dataset because it is useless

a = list(dataset['rate'])
for i in range(0, len(a)):
    if a[i] == 'nan':
        a[i] = 'unrated'
    elif a[i] == '-':
        a[i] = 'unrated'
    elif a[i] == 'NEW':
        a[i] = 'unrated'
        
dataset['rate'] = a

#dataset['rate'].value_counts()
#dataset['rate'] = dataset['rate'].fillna("NEW")
#
#dataset = dataset[dataset['rate'] != 'NEW']
#
#dataset = dataset[dataset['rate'] != "-"]
#
#dataset = dataset[dataset['rate'] != "nan"]

#Slicing and Dicing

X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,-1].values

#handling missing values

test = pd.DataFrame(X)

test[0].isna().sum()
test[1].isna().sum()



test[3].isna().sum()


test[3].isna().sum()
test[4].isna().sum()
test[4].value_counts()

test[5].isna().sum()

test[5].value_counts()

test[5] = test[5].fillna("BTM")

test[5].isna().sum()

test[6].isna().sum()

test[6].value_counts()

test[6] = test[6].fillna("Quick Bites")

test[6].isna().sum()

test[7].isna().sum()
test[7].value_counts()

test[7] = test[7].fillna("Biryani")

test[7].isna().sum()

test[8].isna().sum()

test[8].value_counts()

test[8] = test[8].fillna("North Indian")

test[8].isna().sum()

test[9].isna().sum()

test[9].value_counts()

test[9] = test[9].fillna(300)

test[9].isna().sum()

test[10].isna().sum()

del test[0]

del test[7]

X = test.values

from sklearn.preprocessing import LabelEncoder

lab = LabelEncoder()

X[:,0] = lab.fit_transform(X[:,0])

X[:,1] = lab.fit_transform(X[:,1])

X[:,4] = lab.fit_transform(X[:,4])

X[:,-1] = lab.fit_transform(X[:,-1])

test2 = pd.DataFrame(X)

test2.isnull().sum()

#Handling 5th column which has multiple categorical values in one cell

df = test2

df = df.apply(lambda x: x[5].split(","), axis=1)

from sklearn.preprocessing import MultiLabelBinarizer

mlb = MultiLabelBinarizer()

array_out = mlb.fit_transform(df)

df_out = pd.DataFrame(data=array_out, columns=mlb.classes_)

df_out.to_csv("new.csv",index = False)

#del test2[5]

test2.to_csv("new1.csv")

dataset2 = pd.read_csv("data.csv")

#Handling 6th column which has multiple categorical values in one cell

df = dataset2

df = df.apply(lambda x: x[3].split(","), axis=1)

array_out = mlb.fit_transform(df)

df_out = pd.DataFrame(data=array_out, columns=mlb.classes_)

df_out.to_csv("new111.csv",index = False)

dataset3 = pd.read_csv("data.csv")

X = dataset3.values

test = pd.DataFrame(y)

test[0].value_counts()

# Doing Broad Classification for better classification

a = list(test[0])
for i in range(0, len(a)):
    if a[i] == 'Koramangala 7th Block':
        a[i] = 'Koramangala'
    elif a[i] == 'Koramangala 4th Block':
        a[i] = 'Koramangala'
    elif a[i] == 'Koramangala 5th Block':
        a[i] = 'Koramangala'
    elif a[i] == 'Koramangala 6th Block':
        a[i] = 'Koramangala'
        
test[0] = a

a = list(test[0])
for i in range(0, len(a)):
    if a[i] == 'Bellandur':
        a[i] = 'Bellandur or Frazer Town or Malleshwaram or Rajajinagar or Electronic City or Banashankari or New BEL Road'
    elif a[i] == 'Electronic City':
        a[i] = 'Bellandur or Frazer Town or Malleshwaram or Rajajinagar or Electronic City or Banashankari or New BEL Road'
    elif a[i] == 'Banashankari':
        a[i] = 'Bellandur or Frazer Town or Malleshwaram or Rajajinagar or Electronic City or Banashankari or New BEL Road'
    elif a[i] == 'New BEL Road':
        a[i] = 'Bellandur or Frazer Town or Malleshwaram or Rajajinagar or Electronic City or Banashankari or New BEL Road'
    elif a[i] == 'Rajajinagar':
        a[i] = 'Bellandur or Frazer Town or Malleshwaram or Rajajinagar or Electronic City or Banashankari or New BEL Road'
    elif a[i] == 'Malleshwaram':
        a[i] = 'Bellandur or Frazer Town or Malleshwaram or Rajajinagar or Electronic City or Banashankari or New BEL Road'
    elif a[i] == 'Frazer Town':
        a[i] = 'Bellandur or Frazer Town or Malleshwaram or Rajajinagar or Electronic City or Banashankari or New BEL Road'
        
test[0] = a

test[0].value_counts()

a = list(test[0])
for i in range(0, len(a)):
    if a[i] == 'Brookefield':
        a[i] = 'Brookefield or Basavanagudi or Sarjapur Road or Kammanahalli or Kalyan Nagar'
    elif a[i] == 'Basavanagudi':
        a[i] = 'Brookefield or Basavanagudi or Sarjapur Road or Kammanahalli or Kalyan Nagar'
    elif a[i] == 'Sarjapur Road':
        a[i] = 'Brookefield or Basavanagudi or Sarjapur Road or Kammanahalli or Kalyan Nagar'
    elif a[i] == 'Kammanahalli':
        a[i] = 'Brookefield or Basavanagudi or Sarjapur Road or Kammanahalli or Kalyan Nagar'
    elif a[i] == 'Kalyan Nagar':
        a[i] = 'Brookefield or Basavanagudi or Sarjapur Road or Kammanahalli or Kalyan Nagar'

test[0] = a

test[0].value_counts()

a = list(test[0])
for i in range(0, len(a)):
    if a[i] == 'Old Airport Road':
        a[i] = 'Old Airport Road or Whitefield or Bannerghatta Road or Marathahalli'
    elif a[i] == 'Whitefield':
        a[i] = 'Old Airport Road or Whitefield or Bannerghatta Road or Marathahalli'
    elif a[i] == 'Bannerghatta Road':
        a[i] = 'Old Airport Road or Whitefield or Bannerghatta Road or Marathahalli'
    elif a[i] == 'Marathahalli':
        a[i] = 'Old Airport Road or Whitefield or Bannerghatta Road or Marathahalli'

test[0] = a

a = list(test[0])
for i in range(0, len(a)):
    if a[i] == 'HSR':
        a[i] = 'HSR or Residency Road or Lavelle Road or Brigade Road'
    elif a[i] == 'Residency Road':
        a[i] = 'HSR or Residency Road or Lavelle Road or Brigade Road'
    elif a[i] == 'Lavelle Road':
        a[i] = 'HSR or Residency Road or Lavelle Road or Brigade Road'
    elif a[i] == 'Brigade Road':
        a[i] = 'HSR or Residency Road or Lavelle Road or Brigade Road'

test[0] = a 

a = list(test[0])
for i in range(0, len(a)):
    if a[i] == 'BTM':
        a[i] = 'BTM or Jayanagar or JP Nagar or Indiranagar or MG Road or Church Street'
    elif a[i] == 'Jayanagar':
        a[i] = 'BTM or Jayanagar or JP Nagar or Indiranagar or MG Road or Church Street'
    elif a[i] == 'JP Nagar':
        a[i] = 'BTM or Jayanagar or JP Nagar or Indiranagar or MG Road or Church Street'
    elif a[i] == 'Indiranagar':
        a[i] = 'BTM or Jayanagar or JP Nagar or Indiranagar or MG Road or Church Street'
    elif a[i] == 'MG Road':
        a[i] = 'BTM or Jayanagar or JP Nagar or Indiranagar or MG Road or Church Street'
    elif a[i] == 'Church Street':
        a[i] = 'BTM or Jayanagar or JP Nagar or Indiranagar or MG Road or Church Street'


test[0] = a


y = test[0].values

X = pd.DataFrame(X).values

#Splitting Data into training and testing set

from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state = 100)

#Training Data using Decision Tree Algorithm

from sklearn.tree import DecisionTreeClassifier

dt = DecisionTreeClassifier(max_depth = 19,criterion="gini")

dt.fit(X_train,y_train)

dt.score(X_train,y_train)

dt.score(X_test,y_test)

# Fine tuning

a1 = []

for i in range(1,100):
    dt = DecisionTreeClassifier(max_depth = i,criterion="gini")
    dt.fit(X_train,y_train)
    k = dt.score(X_test,y_test)
    a1.append(k)

