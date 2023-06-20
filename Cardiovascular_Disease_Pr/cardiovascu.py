#!/usr/bin/env python
# coding: utf-8

# # Importing Different Packages Needed For The Project

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier 
import warnings
warnings.filterwarnings('ignore')


# # importing Data

# In[2]:


import pandas as pd
import numpy as np
# Load the data
data = pd.read_csv("cardio_train.csv", delimiter=';')
type(data)


# In[3]:


data.head(5)


# In[4]:


data.shape


# In[5]:


data.sample(5)


# ## cleaning and understanding our dataset

# In[6]:


print(data['cholesterol'].unique())
data['cardio'].nunique()


# In[7]:


data.nunique()


# ### unique values are mainly high in age,height,weight,ap_hi,ap_lo

# In[8]:


data.drop("id",axis=1,inplace=True)


# In[9]:


data.duplicated().sum()


# In[10]:


data.drop_duplicates(inplace=True)


# In[11]:


info = ["age in days","Gender | 1: female, 2: male","Height","Weight","Systolic blood pressure","Diastolic blood pressure","Cholesterol | type-1: normal, 2: above normal, 3: well above normal|","Glucose | type-- 1: normal, 2: above normal, 3: well above normal |","Smoking ","Alcohol intake ","Physical activity","Presence or absence of cardiovascular disease"]



for i in range(len(info)):
    print(data.columns[i]+":\t\t\t"+info[i])


# In[12]:


data.info()


# In[13]:


data.isnull().sum()


# # thankfuly no nan values

# In[14]:


data['cholesterol'].value_counts()


# In[15]:


len(data)


# In[16]:


# Convert age from days to years
data['age'] = (data['age'] / 365).round().astype(int)
data.head(5)


# In[17]:


data['weight'].min()


# In[18]:


data['weight'].max()


# In[19]:


data['height'].min()


# In[20]:


data['height'].max()


# ### check for outliers

# In[21]:


import matplotlib.pyplot as plt
plt.figure(num=None, figsize=(10.4, 6.4), dpi=900, facecolor='w', edgecolor='k')
data.plot(kind='box')
plt.show();


# ### In addition, in some cases diastolic pressure is higher than systolic, which is also incorrect. How many records are inaccurate in terms of blood pressure?

# In[22]:


print("Diastilic pressure is higher than systolic one in {0} cases".format(data[data['ap_lo']> data['ap_hi']].shape[0]))


# In[23]:


data[data['ap_lo']> data['ap_hi']].shape[0]


# ### Let's get rid of the outliers, moreover blood pressure could not be negative value!

# In[24]:


data.drop(data[(data['ap_hi'] > data['ap_hi'].quantile(0.975)) | (data['ap_hi'] < data['ap_hi'].quantile(0.025))].index,inplace=True)
data.drop(data[(data['ap_lo'] > data['ap_lo'].quantile(0.975)) | (data['ap_lo'] < data['ap_lo'].quantile(0.025))].index,inplace=True)


# In[25]:


data[data['ap_lo']> data['ap_hi']].shape[0]


# In[26]:


plt.figure(num=None, figsize=(10.4, 6.4), dpi=900, facecolor='w', edgecolor='k')
data.plot(kind='box')
plt.show();


# ### Let's remove weights and heights, that fall below 2.5% or above 97.5% of a given range.

# In[27]:


data.drop(data[(data['height'] > data['height'].quantile(0.975)) | (data['height'] < data['height'].quantile(0.025))].index,inplace=True)
data.drop(data[(data['weight'] > data['weight'].quantile(0.975)) | (data['weight'] < data['weight'].quantile(0.025))].index,inplace=True)


# In[28]:


(data['height'] > 150).value_counts()


# In[29]:


plt.figure(num=None, figsize=(10.4, 6.4), dpi=900, facecolor='w', edgecolor='k')
data.plot(kind='box')
plt.show();


# In[30]:


data['height'].max()


# In[31]:


data['height'].min()


# In[32]:


data['weight'].max()


# In[33]:


data['weight'].min()


# In[34]:


len(data)


# In[35]:


len(data)


# In[36]:


data['weight'].value_counts()


# In[37]:


print('Maximum age variable:',data["age"].max())
print('Minimum age variable:',data["age"].min())
print(f'Number of age variables:',data["age"].nunique())


# In[38]:


data.info()


# In[39]:


data.describe()


# ### Analysing the 'target' variable

# In[40]:


data["cardio"].describe()


# In[41]:


data["cardio"].unique()


# ##### Clearly, this is a classification problem, with the target variable having values '0' and '1'

# In[42]:


print(data.corr()["cardio"].abs().sort_values(ascending=False))


# In[43]:


#check for correlation among the numerical columns
correlation = data.select_dtypes('number').corr()
correlation


# In[44]:


import seaborn as sns
#visualise the correlation
sns.heatmap(correlation, annot=True, vmin=-1, vmax=1, cmap='Purples', linewidth=0.5);


# # analysing

# ## EDA

# In[45]:




y = data["cardio"]
sns.countplot(y)

target_temp = data.cardio.value_counts()
print(target_temp)


# In[46]:


sns.barplot(data["gender"],y)


# In[47]:


sns.countplot(x='cardio',hue='gender',data=data)


# In[48]:


sns.barplot(data=data,x="cholesterol", y="cardio")
# plt.show()


# In[49]:


value_counts = data.groupby(["cholesterol", "cardio"]).size()

print(value_counts)


# In[50]:


import seaborn as sns
import matplotlib.pyplot as plt

# Calculate the value count of "cardio" for each unique "cholesterol" value
value_counts = data.groupby("cholesterol")["cardio"].value_counts().unstack()

# Plot the value counts
ax = value_counts.plot(kind="bar", stacked=True)

# Customize the plot
ax.set_xlabel("Cholesterol")
ax.set_ylabel("Count")
ax.set_title("Cholesterol with respect to Cardio Count")
plt.legend(title="Cardio", loc="upper right")

# Display the plot
plt.show()


# In[51]:


sns.barplot(data=data,x="gluc", y="cardio")
# plt.show()


# In[52]:


sns.barplot(data=data,x="smoke", y="cardio")
# plt.show()


# In[53]:


sns.barplot(data=data,x="alco", y="cardio")
# plt.show()


# In[54]:


data.groupby('gender')['alco'].sum()


# In[55]:


sns.barplot(data=data,x="active", y="cardio")
# plt.show()


# In[56]:


data['height'].plot.hist()#automaticaly count is taken


# In[57]:


data['weight'].plot.hist()


# In[58]:


data['ap_hi'].plot.hist()


# In[59]:


data['ap_lo'].plot.hist()


# In[60]:


data['BMI'] = data['weight']/((data['height']/100)**2)


# In[61]:


data.head(5)


# In[62]:


data.drop(['height','weight'],axis=1,inplace=True)


# In[63]:


data['BMI'].plot.hist()


# In[64]:


blood_pressure = data.loc[:,['ap_lo','ap_hi']]
sns.boxplot(x = 'variable',y = 'value',data = blood_pressure.melt())
print("Diastilic pressure is higher than systolic one in {0} cases".format(data[data['ap_lo']> data['ap_hi']].shape[0]))


# In[65]:


#Visualize the relationships among age, bmi and avg_glucose_level
columns= ['age', 'BMI', 'gluc']
sns.pairplot(data[columns])
plt.show()


# In[66]:


print(data.corr()["cardio"].abs().sort_values(ascending=False))


# In[67]:


# # One-hot encode categorical variables
# data = pd.get_dummies(data, columns=[ 'cholesterol'])
# data


# In[68]:


data = pd.get_dummies(data, columns=['cholesterol', 'gluc'], drop_first=True)


# In[69]:


data.hist();


# # Train Test split

# In[70]:


# Split into features and labels
x = data.drop(['cardio'], axis=1)
y = data['cardio']


# In[71]:


# Normalize the features
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
x = scaler.fit_transform(x)


# In[72]:


from sklearn.model_selection import train_test_split,GridSearchCV
# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42)


# In[73]:



x_test.shape
x_train.shape


# In[74]:


y_train.shape
y_test.shape


# # logistic regression

# In[75]:


log_reg = LogisticRegression(class_weight='balanced')
param_grid={
    'C': [0.01, 0.1, 1.0, 10, 100],
    'penalty': ['none', 'l1', 'l2', 'elasticnet'],
    'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']
}

#perform grid search with cross-validation to obtain the best hyperparameters
grid_search = GridSearchCV(log_reg, param_grid, cv=5)
grid_search.fit(x_train, y_train)

#print the best hyperparameters
print(grid_search.best_params_)


# In[76]:


#instantiate, fit, and predict with the logistic regression
log_reg = LogisticRegression(C=.01, penalty='l2', solver='liblinear', class_weight='balanced')
log_reg.fit(x_train, y_train)
y_pred_lr = log_reg.predict(x_test)
y_pred_lr


# In[79]:


#check for the accuracy score of the log_reg
score_log_reg = round(accuracy_score(y_test, y_pred_lr)*100,2)
score_log_reg


# In[80]:


print("The accuracy score achieved using Logistic regression is: "+str(score_log_reg)+" %")


# In[81]:


#evaluate and print the train set accuracy
log_reg_train_accuracy = log_reg.score(x_train, y_train)
log_reg_train_accuracy


# In[82]:


#evaluate and print the test set accuracy
log_reg_test_accuracy = log_reg.score(x_test, y_test)
log_reg_test_accuracy


# # SVM

# In[83]:


from sklearn import svm

sv = svm.SVC(kernel='linear')
# from sklearn.svm import SVC
# sv = SVC()
sv.fit(x_train, y_train)

y_pred_svm = sv.predict(x_test)


# In[84]:


y_pred_svm.shape


# In[85]:


score_svm = round(accuracy_score(y_pred_svm,y_test)*100,2)

print("The accuracy score achieved using Linear SVM is: "+str(score_svm)+" %")


# # K Nearest Neighbors

# In[86]:


from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=7)
knn.fit(x_train,y_train)
y_pred_knn=knn.predict(x_test)


# In[87]:


y_pred_knn.shape


# In[88]:


score_knn = round(accuracy_score(y_pred_knn,y_test)*100,2)

print("The accuracy score achieved using KNN is: "+str(score_knn)+" %")


# # Naive Bayes

# In[89]:


from sklearn.naive_bayes import GaussianNB

nb = GaussianNB()

nb.fit(x_train,y_train)

y_pred_nb = nb.predict(x_test)


# In[90]:


y_pred_nb.shape


# In[91]:


score_nb = round(accuracy_score(y_pred_nb,y_test)*100,2)

print("The accuracy score achieved using Naive Bayes is: "+str(score_nb)+" %")


# # Decision Tree

# ### Let's perform hyperparameter tuning for our Decision Tree model:

# In[92]:


dt = DecisionTreeClassifier(class_weight='balanced')
param_grid = {
    'max_depth': [3,4,5,6,7,8],
    'min_samples_split': [2,3,4],
    'min_samples_leaf': [1,2,3,4],
    'random_state':[0, 42]
}

#perform grid search with cross-validation to obtain the best hyperparameters
grid_search = GridSearchCV(dt, param_grid, cv=5)
grid_search.fit(x_train, y_train)

#print the best hyperparameters
print(grid_search.best_params_)


# In[93]:


#instantiate, fit, and predict with DecisionTreeClassifier
dt = DecisionTreeClassifier(max_depth=5, min_samples_leaf=4, min_samples_split=2, random_state=0)
dt.fit(x_train, y_train)
y_pred_dt = dt.predict(x_test)


# In[94]:


#check for the accuracy score of the dt
score_dt = round(accuracy_score(y_test, y_pred_dt)*100,2)
score_dt


# In[95]:


print("The accuracy score achieved using Decision tree is: "+str(score_dt)+" %")


# In[96]:


#evaluate and print the train set accuracy
dt_train_accuracy = dt.score(x_train, y_train)
dt_train_accuracy


# In[97]:


#evaluate and print the test set accuracy
dt_test_accuracy = dt.score(x_test, y_test)
dt_test_accuracy


# # Random Forest

# In[98]:


rfc = RandomForestClassifier(class_weight='balanced')
param_grid={
    'n_estimators': [100, 200],
    'max_depth': [None, 5, 10],
    'max_features': ['sqrt', 'log2', None],
    'random_state': [0, 42]
}

#perform grid search with cross-validation to obtain the best hyperparameters
grid_search = GridSearchCV(rfc, param_grid, cv=5)
grid_search.fit(x_train, y_train)

#print the best hyperparameters
print(grid_search.best_params_)


# In[99]:


#instantiate, fit, and predict with Random Forest Classifier
rfc = RandomForestClassifier(max_depth=10, max_features='sqrt', n_estimators=200, random_state=0, class_weight='balanced')
rfc.fit(x_train, y_train)
y_pred_rfc = rfc.predict(x_test)


# In[100]:


#check for the accuracy score of the dt
score_rfc = round(accuracy_score(y_test, y_pred_rfc)*100,2)
score_rfc


# In[101]:


print("The accuracy score achieved using RandomForest is: "+str(score_rfc)+" %")


# In[102]:


#evaluate and print the train set accuracy
rfc_train_accuracy = rfc.score(x_train, y_train)
rfc_train_accuracy


# In[103]:


#evaluate and print the test set accuracy
rfc_test_accuracy = rfc.score(x_test, y_test)
rfc_test_accuracy


# ## Now, let's compare the accuracy of the three models we've deployed:

# In[116]:


scores = [score_log_reg,score_nb,score_svm,score_knn,score_dt,score_rfc]
algorithms = ["Logistic Regression","Naive Bayes","Support Vector Machine","K-Nearest Neighbors","Decision Tree","Random Forest"]    

for i in range(len(algorithms)):
    print("The accuracy score achieved using "+algorithms[i]+" is: "+str(scores[i])+" %")


# In[117]:


sns.set(rc={'figure.figsize':(15,8)})
plt.xlabel("Algorithms")
plt.ylabel("Accuracy score")

sns.barplot(algorithms,scores)


# ### Hey there random forest has good result as compare to other algorithms <br> <br>
