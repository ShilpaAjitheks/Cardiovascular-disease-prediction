#!/usr/bin/env python
# coding: utf-8

# # Importing Different Packages Needed For The Project

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
# plt.rcParams['figure.figsize'] = (12, 12)
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier 
from sklearn.naive_bayes import GaussianNB
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
import warnings
warnings.filterwarnings('ignore')


# # importing Data

# In[2]:


# Load the data
data = pd.read_csv("cardio_train.csv", delimiter=';')
type(data)


# In[3]:


data.sample(5)


# In[4]:


data.shape


# ## cleaning and understanding our dataset

# In[5]:


print(data['cholesterol'].unique())
data['cardio'].nunique()


# In[6]:


data.nunique()


# ### unique values are mainly high in columns age,height,weight,ap_hi,ap_lo

# In[7]:


#lets drop id as it is not relavent
data.drop("id",axis=1,inplace=True)


# In[8]:


# Convert age from days to years
data['age'] = (data['age'] / 365).round().astype(int)
data.head(5)


# ### later we will deal with numerical values in height,weight,ap_hi,ap_lo columns

# In[9]:


data.duplicated().sum()


# In[10]:


data.drop_duplicates(inplace=True)


# In[11]:


info = ["age in days","Gender | 1: female, 2: male","Height-(cm)","Weight-(kg)","Systolic blood pressure-(norm:120 hg)","Diastolic blood pressure-(norm:80 hg)","Cholesterol | type-1: normal, 2: above normal, 3: well above normal|","Glucose | type-- 1: normal, 2: above normal, 3: well above normal |","Smoking |type- 0: do 1: dont do","Alcohol intake|type- 0: do 1: dont do ","Physical activity |type- 0: do 1: dont do","Presence or absence of cardiovascular disease"]



for i in range(len(info)):
    print(data.columns[i]+":\t\t\t"+info[i])


# In[12]:


data.info()


# In[13]:


data.isnull().sum()


# # thankfuly no nan values

# In[14]:


data['age'].nunique()


# In[15]:


data['cholesterol'].value_counts()


# In[16]:


data['gluc'].value_counts()


# In[17]:


data["bmi"] = data["weight"] / (data["height"]/100)**2


# In[18]:


data.drop(["weight","height"],axis=1,inplace=True)


# ### check for outliers

# In[19]:


plt.figure(num=None, figsize=(10.4, 6.4), dpi=900, facecolor='w', edgecolor='k')
data.plot(kind='box')
plt.axis(xmin=0,xmax=14,ymin=1,ymax=200)
plt.show();


# ### In  some cases diastolic pressure is higher than systolic, which is  incorrect. How many records are inaccurate in terms of blood pressure?

# In[20]:


print("Diastolic pressure is higher than systolic one in {0} cases".format(data[data['ap_lo']> data['ap_hi']].shape[0]))


# ### Let's get rid of the outliers, moreover blood pressure could not be negative value!

# In[21]:


print('Maximum systolic pressure:',data["ap_hi"].max())
print('Minimum systolic pressure:', data["ap_hi"].min())
print('Number of systolic pressure variables:', data["ap_hi"].nunique())


# In[22]:


print('Maximum diastolic pressure:',data["ap_lo"].max())
print('Minimum diastolic pressure:', data["ap_lo"].min())
print('Number of diastolic pressure variables:', data["ap_lo"].nunique())


# In[23]:


# data.drop(data[(data['ap_hi'] > data['ap_hi'].quantile(0.975)) | (data['ap_hi'] < data['ap_hi'].quantile(0.025))].index,inplace=True)
# data.drop(data[(data['ap_lo'] > data['ap_lo'].quantile(0.975)) | (data['ap_lo'] < data['ap_lo'].quantile(0.025))].index,inplace=True)


# In[24]:


# out_filter = ((data["ap_hi"]>250) | (data["ap_lo"]>200))
out_filter = ((data["ap_hi"]>175) | (data["ap_lo"]>120))
data = data[~out_filter]
len(data)


# In[25]:


# out_filter2 = ((data["ap_hi"] < 0) | (data["ap_lo"] < 0))
out_filter2 = ((data["ap_hi"] < 75) | (data["ap_lo"] < 50))
data = data[~out_filter2]
len(data)


# In[26]:


data[data['ap_lo']> data['ap_hi']].shape[0]


# In[27]:


#Collapse ap_hi into fewer groups
ranges = [0, 130, 180, 320]
group_names = ['Normal', 'Hypertension', 'Hypertensive crisis']
data['systolic'] = pd.cut(data['ap_hi'], bins=ranges, labels=group_names)
data['systolic'].unique()


# In[28]:


#Collapse ap_lo into fewer groups
ranges = [-1, 81, 120, 201]
group_names = ['Normal', 'Hypertension', 'Hypertensive crisis']
data['diastolic'] = pd.cut(data['ap_lo'], bins=ranges, labels=group_names)
data['diastolic'].unique()


# ### Let's remove bmi

# In[29]:


print('Maximum body mass index:',data["bmi"].max())
print('Minimum body mass index:', data["bmi"].min())
print('Number of body mass index:', data["bmi"].nunique())


# In[30]:


# out_filter2 = ((data["bmi"]>150))
out_filter2 = ((data["bmi"]>50))
data = data[~out_filter2]
len(data)


# In[31]:


#Collapse bmi into fewer groups
ranges = [0, 19, 25, 30, 160]
group_names = ['Underweight', 'Normal', 'Overweight', 'Obesity']
data['bmi_group'] = pd.cut(data['bmi'], bins=ranges, labels=group_names)
data['bmi_group'].unique()


# In[32]:


# plt.figure(num=None, figsize=(10.4, 6.4), dpi=900, facecolor='w', edgecolor='k')
data.plot(kind='box')
# plt.axis(xmin=0,xmax=14,ymin=-1,ymax=5)
# plt.show();


# In[33]:


data.info()


# In[34]:


data.describe()


# ### Analysing the 'target' variable

# In[35]:


data["cardio"].describe()


# In[36]:


data["cardio"].unique()


# ##### Clearly, this is a classification problem, with the target variable having values '0' and '1'

# In[37]:


print(data.corr()["cardio"].abs().sort_values(ascending=False))


# In[38]:


#check for correlation among the numerical columns
correlation = data.select_dtypes('number').corr()
correlation


# In[39]:


import seaborn as sns
#visualise the correlation
sns.heatmap(correlation, annot=True, vmin=-1, vmax=1, cmap='Purples', linewidth=0.5);


# #### here we can see ap_hi and ap_lo are little bit dependent
# #### gluc and cholesterol are also dependent

# # analysing

# ## EDA

# In[40]:


y = data["cardio"]
sns.countplot(y)

target_temp = data.cardio.value_counts()
print(target_temp)


# In[41]:


sns.barplot(data["gender"],y)


# ##### avg value of y for each men and women is ploted

# In[42]:


sns.countplot(x='cardio',hue='gender',data=data)


# ##### count of occurance of each category

# In[43]:


sns.barplot(data=data,x="cholesterol", y="cardio")
# plt.show()


# In[44]:


value_counts = data.groupby(["cholesterol", "cardio"]).size()

print(value_counts)


# In[45]:


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


# In[46]:


sns.barplot(data=data,x="gluc", y="cardio")
# plt.show()


# In[47]:


sns.barplot(data=data,x="smoke", y="cardio")
# plt.show()


# In[48]:


sns.barplot(data=data,x="alco", y="cardio")
# plt.show()


# In[49]:


data.groupby('gender')['alco'].sum()


# In[50]:


sns.barplot(data=data,x="active", y="cardio")
# plt.show()


# In[51]:


data['ap_hi'].plot.hist()


# In[52]:


data['ap_lo'].plot.hist()


# In[53]:


data['bmi'].plot.hist()


# In[54]:


blood_pressure = data.loc[:,['ap_lo','ap_hi']]
sns.boxplot(x = 'variable',y = 'value',data = blood_pressure.melt())
print("Diastilic pressure is higher than systolic one in {0} cases".format(data[data['ap_lo']> data['ap_hi']].shape[0]))


# In[55]:


#Visualize the relationships among age, bmi and avg_glucose_level
columns= ['age', 'bmi', 'cholesterol']
sns.pairplot(data[columns])
plt.show()


# In[56]:


print(data.corr()["cardio"].abs().sort_values(ascending=False))


# In[57]:


columns = ['bmi_group', 'systolic','diastolic', 'cholesterol','gluc','gender']

for column in columns:
    unique_values = data[column].unique()
    print(f"Unique values for {column}:{unique_values}")


# ## Binary Encoding:

# In[58]:


labelencoder = LabelEncoder()
data['gender']=labelencoder.fit_transform(data['gender'])


# In[59]:


data.head()


# ## Label Encoding:

# In[60]:


#Encode for categorical columns
cat_cols = ['systolic', 'diastolic', 'bmi_group']

for col in cat_cols:
    data[col] = labelencoder.fit_transform(data[col])


# In[61]:


data.sample(5)


# ## One-Hot-Encoding:

# In[62]:


data = pd.get_dummies(data, columns=['cholesterol', 'gluc','systolic', 'diastolic', 'bmi_group'], drop_first=True)


# In[63]:


data.sample(5)


# In[64]:


data.hist();


# # Train Test split

# In[65]:


# Split into features and labels
x = data.drop(['cardio'], axis=1)
y = data['cardio']


# In[66]:


# Normalize the features
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
x = scaler.fit_transform(x)


# In[67]:


from sklearn.model_selection import train_test_split,GridSearchCV
# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42)


# In[68]:



x_test.shape
x_train.shape


# In[69]:


y_train.shape
y_test.shape


# # logistic regression

# In[70]:


log_reg = LogisticRegression(class_weight='balanced')
param_grid={
    'C': [0.01, 0.1, 1.0, 10, 100],
    'penalty': ['none', 'l1', 'l2', 'elasticnet'],
    'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']
}

# perform grid search with cross-validation to obtain the best hyperparameters
grid_search = GridSearchCV(log_reg, param_grid, cv=5)
grid_search.fit(x_train, y_train)

#print the best hyperparameters
print(grid_search.best_params_)


# In[71]:


#instantiate, fit, and predict with the logistic regression
log_reg = LogisticRegression(C=.01, penalty='l2', solver='liblinear', class_weight='balanced')
log_reg.fit(x_train, y_train)
y_pred_lr = log_reg.predict(x_test)
y_pred_lr


# In[72]:


#check for the accuracy score of the log_reg
score_log_reg = round(accuracy_score(y_test, y_pred_lr)*100,2)
score_log_reg


# In[73]:


print("The accuracy score achieved using Logistic regression is: "+str(score_log_reg)+" %")


# In[74]:


#evaluate and print the train set accuracy
log_reg_train_accuracy = log_reg.score(x_train, y_train)
log_reg_train_accuracy


# In[75]:


#evaluate and print the test set accuracy
log_reg_test_accuracy = log_reg.score(x_test, y_test)
log_reg_test_accuracy


# # SVM

# In[76]:


from sklearn import svm

sv = svm.SVC(kernel='linear')
# from sklearn.svm import SVC
# sv = SVC()
sv.fit(x_train, y_train)

y_pred_svm = sv.predict(x_test)


# In[77]:


y_pred_svm.shape


# In[78]:


score_svm = round(accuracy_score(y_pred_svm,y_test)*100,2)

print("The accuracy score achieved using Linear SVM is: "+str(score_svm)+" %")


# In[79]:


#evaluate and print the train set accuracy
svm_train_accuracy = sv.score(x_train, y_train)
svm_train_accuracy


# In[80]:


#evaluate and print the test set accuracy
svm_test_accuracy = sv.score(x_test, y_test)
svm_test_accuracy


# # K Nearest Neighbors

# In[81]:


from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=7)
knn.fit(x_train,y_train)
y_pred_knn=knn.predict(x_test)


# In[82]:


y_pred_knn.shape


# In[83]:


score_knn = round(accuracy_score(y_pred_knn,y_test)*100,2)

print("The accuracy score achieved using KNN is: "+str(score_knn)+" %")


# In[84]:


#evaluate and print the train set accuracy
knn_train_accuracy = knn.score(x_train, y_train)
knn_train_accuracy


# In[85]:


#evaluate and print the test set accuracy
knn_test_accuracy = knn.score(x_test, y_test)
knn_test_accuracy


# # Naive Bayes

# In[86]:


from sklearn.naive_bayes import GaussianNB

nb = GaussianNB()

nb.fit(x_train,y_train)

y_pred_nb = nb.predict(x_test)


# In[87]:


y_pred_nb.shape


# In[88]:


score_nb = round(accuracy_score(y_pred_nb,y_test)*100,2)

print("The accuracy score achieved using Naive Bayes is: "+str(score_nb)+" %")


# In[89]:


#evaluate and print the train set accuracy
nb_train_accuracy = nb.score(x_train, y_train)
nb_train_accuracy


# In[90]:


#evaluate and print the test set accuracy
nb_test_accuracy = nb.score(x_test, y_test)
nb_test_accuracy


# # Decision Tree

# ### Let's perform hyperparameter tuning for our Decision Tree model:

# In[91]:


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


# In[92]:


#instantiate, fit, and predict with DecisionTreeClassifier
dt = DecisionTreeClassifier(max_depth=3, min_samples_leaf=4, min_samples_split=2, random_state=0)
dt.fit(x_train, y_train)
y_pred_dt = dt.predict(x_test)


# In[93]:


#check for the accuracy score of the dt
score_dt = round(accuracy_score(y_test, y_pred_dt)*100,2)
score_dt


# In[94]:


print("The accuracy score achieved using Decision tree is: "+str(score_dt)+" %")


# In[95]:


#evaluate and print the train set accuracy
dt_train_accuracy = dt.score(x_train, y_train)
dt_train_accuracy


# In[96]:


#evaluate and print the test set accuracy
dt_test_accuracy = dt.score(x_test, y_test)
dt_test_accuracy


# # Random Forest

# In[97]:


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


# In[98]:


#instantiate, fit, and predict with Random Forest Classifier
rfc = RandomForestClassifier(max_depth=10, max_features='sqrt', n_estimators=200, random_state=42, class_weight='balanced')
rfc.fit(x_train, y_train)
y_pred_rfc = rfc.predict(x_test)


# In[99]:


#check for the accuracy score of the dt
score_rfc = round(accuracy_score(y_test, y_pred_rfc)*100,2)
score_rfc


# In[100]:


print("The accuracy score achieved using RandomForest is: "+str(score_rfc)+" %")


# In[101]:


#evaluate and print the train set accuracy
rfc_train_accuracy = rfc.score(x_train, y_train)
rfc_train_accuracy


# In[102]:


#evaluate and print the test set accuracy
rfc_test_accuracy = rfc.score(x_test, y_test)
rfc_test_accuracy


# ## Now, let's compare the accuracy of the three models we've deployed:

# In[103]:


scores = [score_log_reg,score_nb,score_svm,score_knn,score_dt,score_rfc]
algorithms = ["Logistic Regression","Naive Bayes","Support Vector Machine","K-Nearest Neighbors","Decision Tree","Random Forest"]    

for i in range(len(algorithms)):
    print("The accuracy score achieved using "+algorithms[i]+" is: "+str(scores[i])+" %")


# In[104]:


sns.set(rc={'figure.figsize':(15,8)})
plt.xlabel("Algorithms")
plt.ylabel("Accuracy score")

sns.barplot(algorithms,scores)


# In[105]:


models = pd.DataFrame({
    'Algorithms' : ["Logistic Regression","Naive Bayes","Support Vector Machine","K-Nearest Neighbors","Decision Tree","Random Forest"]    ,
    'Score_train': [log_reg_train_accuracy,nb_train_accuracy,svm_train_accuracy,knn_train_accuracy,dt_train_accuracy,rfc_train_accuracy],
    'Score_test': [log_reg_test_accuracy,nb_test_accuracy,svm_test_accuracy,knn_test_accuracy,dt_test_accuracy,rfc_test_accuracy],
                    })


# In[106]:


models.sort_values(by=['Score_train', 'Score_test'], ascending=False)


# ### Hey there random forest has good result as compare to other algorithms

# ### Model should have better test_score than train score,then only it is considered as good  <br> <br>
