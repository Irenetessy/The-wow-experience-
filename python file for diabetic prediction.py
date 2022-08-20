#!/usr/bin/env python
# coding: utf-8

# In[226]:


#importing the liberies needed
import pandas as pd #used for data cleaning
import numpy as np
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt #used for data visualization
import seaborn as sns  #used for data visualization
from IPython.display import display
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import classification_report
from sklearn.neural_network import MLPClassifier

import plotly.tools as tls
import plotly.offline as py
py.init_notebook_mode(connected=True)
from scipy import stats
from scipy.stats import norm

plt.style.use('ggplot')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[227]:


#loading the dataset
df = pd.read_csv("diabetes.csv")


# In[228]:


#Feature information,which gives details of each colum in the data set
df.info()


# In[229]:


# The size of the data set was examined. It consists of 768 observation units and 9 variables.
df.shape


# # ANALYSING THE DATA

# In[230]:


df.head()


# In[231]:


# Descriptive statistics of the data set
df.describe()


# In[232]:


#checking for null values in the columns
df.isnull().sum()


# In[233]:


#checking for null values in the dataset
df.isnull().any()


# In[234]:


#checked for null values in the dataset and no null value was present 
df.isnull()


# In[235]:


#Access to the correlation of the data set was provided. What kind of relationship is examined between the variables. 
# If the correlation value is> 0, there is a positive correlation. While the value of one variable increases, the value of the other variable also increases.
# Correlation = 0 means no correlation.
# If the correlation is <0, there is a negative correlation. While one variable increases, the other variable decreases. 
# When the correlations are examined, there are 2 variables that act as a positive correlation to the Salary dependent variable.
# These variables are Glucose. As these increase, Outcome variable increases.
df.corr()


# In[236]:


# Correlation matrix graph of the data set
k = 9
cols = df.corr().nlargest(k, 'Outcome')['Outcome'].index
cm = df[cols].corr()
plt.figure(figsize=(10,6))
sns.heatmap(cm, annot=True, cmap = 'magma')


# In[237]:


#The diagonal shows the distribution of the the dataset with the kernel density plots,The scatter-plots shows the relationship between each and every attribute or features taken pairwise. Looking at the scatter-plots, we can say that no two attributes are able to clearly seperate the two outcome
g = sns.pairplot(df, hue="Outcome", palette="husl")


# In[238]:


#histogram to have a clearer visual of the dataset and see where the outlier are opresent
p = df.hist(figsize = (20,20))


# # Checking for outliers

# In[239]:


# In the data set, there were asked whether there were any outlier observations compared to the 25% and 75% quarters.
# It was found to be an outlier observation.
for feature in df:
    
    Q1 = df[feature].quantile(0.25)
    Q3 = df[feature].quantile(0.75)
    IQR = Q3-Q1
    lower = Q1- 1.5*IQR
    upper = Q3 + 1.5*IQR
    
    if df[(df[feature] > upper)].any(axis=None):
        print(feature,"yes")
    else:
        print(feature, "no")


# In[240]:


# The process of visualizing the Insulin variable with boxplot method was done. We find the outlier observations on the chart.
sns.boxplot(x = df["Insulin"]);


# In[241]:


#We conduct a stand alone observation review for the Insulin variable
#We suppress contradictory values
Q1 = df.Insulin.quantile(0.25)
Q3 = df.Insulin.quantile(0.75)
IQR = Q3-Q1
lower = Q1 - 1.5*IQR
upper = Q3 + 1.5*IQR
df.loc[df["Insulin"] > upper,"Insulin"] = upper
sns.boxplot(x = df["Insulin"]);


# In[242]:


# The process of visualizing the pregnancies variable with boxplot method was done. We find the outlier observations on the chart.
sns.boxplot(x = df["Pregnancies"]);


# In[243]:


#We conduct a stand alone observation review for the pregnancies variable
#We suppress contradictory values
Q1 = df.Pregnancies.quantile(0.25)
Q3 = df.Pregnancies.quantile(0.75)
IQR = Q3-Q1
lower = Q1 - 1.5*IQR
upper = Q3 + 1.5*IQR
df.loc[df["Pregnancies"] > upper,"Pregnancies"] = upper
sns.boxplot(x = df["Pregnancies"]);


# In[244]:


# Visualizing the Skinthickness variable with boxplot method was done. We find the outlier observations on the chart.
sns.boxplot(x = df["SkinThickness"]);


# In[245]:


#conducted a stand alone observation review for the pregnancies variable to suppress contradictory values
Q1 = df.SkinThickness.quantile(0.25)
Q3 = df.SkinThickness.quantile(0.75)
IQR = Q3-Q1
lower = Q1 - 1.5*IQR
upper = Q3 + 1.5*IQR
df.loc[df["SkinThickness"] > upper,"SkinThickness"] = upper
sns.boxplot(x = df["SkinThickness"]);


# In[246]:


#The process of visualizing the BMI variable with boxplot method was done. We find the outlier observations on the chart
sns.boxplot(x = df["BMI"]);


# In[247]:


#We conduct a stand alone observation review for the Insulin variable
#We suppress contradictory values
Q1 = df.BMI.quantile(0.25)
Q3 = df.BMI.quantile(0.75)
IQR = Q3-Q1
lower = Q1 - 1.5*IQR
upper = Q3 + 1.5*IQR
df.loc[df["BMI"] > upper,"BMI"] = upper
sns.boxplot(x = df["BMI"]);


# In[248]:


#The process of visualizing diabetes pedegree function variable with boxplot method was done. We find the outlier observations on the chart
sns.boxplot(x = df["DiabetesPedigreeFunction"]);


# In[249]:


#conducted a stand alone observation review for diabetes pedegree function variable to suppress contradictory values
Q1 = df.DiabetesPedigreeFunction.quantile(0.25)
Q3 = df.DiabetesPedigreeFunction.quantile(0.75)
IQR = Q3-Q1
lower = Q1 - 1.5*IQR
upper = Q3 + 1.5*IQR
df.loc[df["DiabetesPedigreeFunction"] > upper,"DiabetesPedigreeFunction"] = upper
sns.boxplot(x = df["DiabetesPedigreeFunction"]);


# In[250]:


#The process of visualizing the age variable with boxplot method was done. We find the outlier observations on the chart
sns.boxplot(x = df["Age"]);


# In[251]:


#conducted a stand alone observation review for diabetes pedegree function variable to suppress contradictory values

Q1 = df.Age.quantile(0.25)# Calculate Q1 (25th percentile of the data) for the given feature
Q3 = df.Age.quantile(0.75)# Calculate Q3 (75th percentile of the data) for the given feature
IQR = Q3-Q1# Use the interquartile range to calculate an outlier step (1.5 times the interquartile range)
lower = Q1 - 1.5*IQR  # Use the interquartile range to calculate an outlier step (1.5 times the interquartile range)
upper = Q3 + 1.5*IQR
#print and remove outliers
df.loc[df["Age"] > upper,"Age"] = upper
sns.boxplot(x = df["Age"]);


# In[252]:


#The process of visualizing the age variable with boxplot method was done. We find the outlier observations on the chart
sns.boxplot(x = df["BloodPressure"]);


# In[253]:


#conducted a stand alone observation review for diabetes pedegree function variable to suppress contradictory values

Q1 = df.BloodPressure.quantile(0.25)# Calculate Q1 (25th percentile of the data) for the given feature
Q3 = df.BloodPressure.quantile(0.75)# Calculate Q3 (75th percentile of the data) for the given feature
IQR = Q3-Q1
lower = Q1 - 1.5*IQR
upper = Q3 + 1.5*IQR
df.loc[df["BloodPressure"] > upper,"BloodPressure"] = upper
sns.boxplot(x = df["BloodPressure"]);


# In[264]:


for feature in df:
    
    Q1 = df[feature].quantile(0.25)
    Q3 = df[feature].quantile(0.75)
    IQR = Q3-Q1
    lower = Q1- 1.5*IQR
    upper = Q3 + 1.5*IQR
    
    if df[(df[feature] > upper)].any(axis=None):
        print(feature,"yes")
    else:
        print(feature, "no")


# # Feature Engineering and Selection

# In[265]:


# According to BMI, some ranges were determined and categorical variables were assigned.
NewBMI = pd.Series(["Underweight", "Normal", "Overweight", "Obesity 1", "Obesity 2", "Obesity 3"], dtype = "category")
df["NewBMI"] = NewBMI
df.loc[df["BMI"] < 18.5, "NewBMI"] = NewBMI[0]
df.loc[(df["BMI"] > 18.5) & (df["BMI"] <= 24.9), "NewBMI"] = NewBMI[1]
df.loc[(df["BMI"] > 24.9) & (df["BMI"] <= 29.9), "NewBMI"] = NewBMI[2]
df.loc[(df["BMI"] > 29.9) & (df["BMI"] <= 34.9), "NewBMI"] = NewBMI[3]
df.loc[(df["BMI"] > 34.9) & (df["BMI"] <= 39.9), "NewBMI"] = NewBMI[4]
df.loc[df["BMI"] > 39.9 ,"NewBMI"] = NewBMI[5]


# In[266]:


# A categorical variable creation process is performed according to the insulin value.
def set_insulin(row):
    if row["Insulin"] >= 16 and row["Insulin"] <= 166:
        return "Normal"
    else:
        return "Abnormal"


# In[267]:


# The operation performed was added to the dataframe.
df = df.assign(NewInsulinScore=df.apply(set_insulin, axis=1))

df.head()


# # LABEL ENCODING

# In[268]:


from sklearn.preprocessing import LabelEncoder

NewInsulinScore_encoder = LabelEncoder()
df = df.copy()
df.NewInsulinScore = NewInsulinScore_encoder.fit_transform(df.NewInsulinScore)
df

NewBMI_encoder = LabelEncoder()
df = df.copy()
df.NewBMI = NewBMI_encoder.fit_transform(df.NewBMI)
df


# In[269]:


df.shape #showing new arritributes


# In[270]:


df[cols] = df[cols].apply(LabelEncoder().fit_transform)


# In[271]:


#train_test_splitting of the dataset

X=df.drop('NewInsulinScore',1)
Y=df['NewInsulinScore']
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.30,random_state=0)


# In[272]:


#defining evaluating models
data_scores=[]
def EvaluatingModels(true_value,predicted_value,model):
    MSE=mean_squared_error(true_value,predicted_value,squared=True)
    RMSE=mean_squared_error(true_value,predicted_value,squared=False)
    MAE=mean_absolute_error(true_value,predicted_value)
    R_Squared=r2_score(true_value,predicted_value)
    data_scores.append([model,MSE,RMSE,MAE,R_Squared])
    print("MAE_:", MAE)
    print("RMSE_:", RMSE)
    print("MSE_:", MSE)
    print("R_Squared_:", R_Squared)


# In[273]:


#logistic regression
regress_dtree=LogisticRegression()
dtree_model=regress_dtree.fit(X_train,Y_train)
y_predtree=regress_dtree.predict(X_test)
EvaluatingModels(Y_test,y_predtree, 'Logistic Regression')


# In[274]:


#LinearDiscriminantAnalysis
egress_dtree=LinearDiscriminantAnalysis()
dtree_model=regress_dtree.fit(X_train,Y_train)
y_predtree=regress_dtree.predict(X_test)
EvaluatingModels(Y_test,y_predtree, 'Linear Discriminant Analysis')


# In[275]:


#Linear regression
linear_regressor=LinearRegression(normalize=True)
linear_regressor.fit(X_train,Y_train)
lin_prediction=linear_regressor.predict(X_test)
EvaluatingModels(Y_test,lin_prediction, 'Linear Regression')


# In[276]:


df=pd.DataFrame(data_scores,columns=['Model','MSE','RMSE','MAE','R2'])
df


# Linear regression can be used as baseline model as it has the least value

# In[286]:


#support vector classifier(SVC)
svm = SVC()
svm.fit(X_train,Y_train)    
svm_acc= accuracy_score(Y_test,svm.predict(X_test))
EvaluatingModels(Y_test,y_predtree, 'Support Vector Machine')


# In[287]:


regress_dtree=KNeighborsClassifier()
dtree_model =regress_dtree.fit(X_train,Y_train)
y_pretree = regress_dtree.predict(X_test)
EvaluatingModels(Y_test,y_predtree, 'KNeighbors Classifier')


# In[288]:


regress_dtree=DecisionTreeClassifier(criterion='entropy',random_state=0)
dtree_model =regress_dtree.fit(X_train,Y_train)
y_pretree = regress_dtree.predict(X_test)
EvaluatingModels(Y_test,y_predtree, 'Decision Tree Classifier')


# In[289]:


#Naive Bayes
rf_regressor=GaussianNB()
rf_model=rf_regressor.fit(X_train,Y_train)
y_rfpred=rf_model.predict(X_test)

EvaluatingModels(Y_test,y_rfpred, 'NAIVE BAYES')


# In[290]:


# Gradient Boosting Classifier
gbc = GradientBoostingClassifier()
gb = GradientBoostingClassifier(random_state=0,learning_rate=0.01)
gbc.fit(X_train, Y_train)
gbc_acc=accuracy_score(Y_test,gbc.predict(X_test))
EvaluatingModels(Y_test,y_predtree, 'GradientBoostingClassifier')


# In[291]:


#Ada boostclassifier
adb = AdaBoostClassifier(base_estimator = None)
adb.fit(X_train,Y_train)

EvaluatingModels(Y_test,y_rfpred, 'AdaBoostClassifier')


# In[293]:


#Extra Trees Classifier
etc = ExtraTreesClassifier(n_estimators=100, random_state=1)
etc.fit(X_train,Y_train)

EvaluatingModels(Y_test,y_rfpred, 'ExtraTreesClassifier')


# In[294]:


rf_regressor=RandomForestClassifier(random_state=1)
rf_model=rf_regressor.fit(X_train,Y_train)
y_rfpred = rf_model.predict(X_test)
EvaluatingModels(Y_test,y_rfpred, 'Random Forest Classifier')


# Random forest had 100% performance,while gradient boosting classifier,Decision tree classifier,linear regression,Kneighbors,logistic regression had 98% performance 

# In[295]:


df=pd.DataFrame(data_scores,columns=['Model','MSE','RMSE','MAE','R2'])
df


# In[298]:


#scaled and applied kfold to algorithms to reavaluate them
from sklearn.preprocessing import MinMaxScaler
scaler =MinMaxScaler(feature_range=(0, 1))
scaledX =scaler.fit_transform(X_train)


# In[300]:


from sklearn.model_selection import RepeatedStratifiedKFold
models=[] #to initialze the models
models.append(('KNN',KNeighborsClassifier()))
models.append(('SVM',SVC()))
models.append(('DTR',DecisionTreeClassifier(criterion='entropy',random_state=0)))
models.append(('RFR',RandomForestClassifier()))
models.append(('GBC',GradientBoostingClassifier()))
#models are evaluated 
results =[]
names =[]
for name,model in models:
    kfold = RepeatedStratifiedKFold(n_splits=10,n_repeats=3,random_state=1) #defining kfolds to use
    cv_results = cross_val_score(model,scaledX,Y_train,cv=kfold,scoring='accuracy')
    results.append(cv_results)
    names.append(name)
    d_score= print('%s:%f(%f)'% (name,cv_results.mean(),cv_results.std()))
    


# In[302]:


import timeit #to check the time complexity needed to run the algorthim
start=timeit.timeit()
model = RandomForestClassifier()
model.fit(scaledX,Y_train)
#estimated the accuracy on validation dataset
rescaledValidationX = scaler.transform(X_test)
predictions=model.predict(rescaledValidationX)
print(accuracy_score(Y_test, predictions))
print(confusion_matrix(Y_test, predictions))#used tp check confusion matrix
print(classification_report(Y_test, predictions))
end = timeit.timeit()
print(end - start )


# In[305]:


model = DecisionTreeClassifier(criterion='entropy',random_state=0)
model.fit(scaledX,Y_train)
#estimated the accuracy on validation dataset
rescaledValidationX = scaler.transform(X_test) #scaled data
predictions=model.predict(rescaledValidationX)
print(accuracy_score(Y_test, predictions))
print(confusion_matrix(Y_test, predictions))
print(classification_report(Y_test, predictions))
end = timeit.timeit()
print(end - start )#print time complexity


# In[306]:


model =KNeighborsClassifier()
model.fit(scaledX,Y_train)
#estimated the accuracy on validation dataset
rescaledValidationX = scaler.transform(X_test)
predictions=model.predict(rescaledValidationX)
print(accuracy_score(Y_test, predictions))
print(confusion_matrix(Y_test, predictions))
print(classification_report(Y_test, predictions))
end = timeit.timeit()
print(end - start )


# In[307]:


model=GradientBoostingClassifier()
model.fit(scaledX,Y_train)
#estimated the accuracy on validation dataset
rescaledValidationX = scaler.transform(X_test)
predictions=model.predict(rescaledValidationX)
print(accuracy_score(Y_test, predictions))
print(confusion_matrix(Y_test, predictions))
print(classification_report(Y_test, predictions))
end = timeit.timeit()
print(end - start )


# In[313]:



from sklearn import metrics
#set up plotting area
plt.figure(0).clf()

#fit logistic regression model and plot ROC curve
model = LogisticRegression()
model.fit(X_train, Y_train)
y_pred = model.predict_proba(X_test)[:, 1]
fpr, tpr, _ = metrics.roc_curve(Y_test, y_pred)
auc = round(metrics.roc_auc_score(Y_test, y_pred), 4)
plt.plot(fpr,tpr,label="Logistic Regression, AUC="+str(auc))

#fit gradient boosted model and plot ROC curve
model = GradientBoostingClassifier()
model.fit(X_train, Y_train)
y_pred = model.predict_proba(X_test)[:, 1]
fpr, tpr, _ = metrics.roc_curve(Y_test, y_pred)
auc = round(metrics.roc_auc_score(Y_test, y_pred), 4)
plt.plot(fpr,tpr,label="Gradient Boosting, AUC="+str(auc))

#fit random forest classifier model and plot ROC curve
model = RandomForestClassifier()
model.fit(X_train, Y_train)
y_pred = model.predict_proba(X_test)[:, 1]
fpr, tpr, _ = metrics.roc_curve(Y_test, y_pred)
auc = round(metrics.roc_auc_score(Y_test, y_pred), 4)
plt.plot(fpr,tpr,label="RandomForestClassifier, AUC="+str(auc))

#fit random forest classifier model and plot ROC curve
model = DecisionTreeClassifier()
model.fit(X_train, Y_train)
y_pred = model.predict_proba(X_test)[:, 1]
fpr, tpr, _ = metrics.roc_curve(Y_test, y_pred)
auc = round(metrics.roc_auc_score(Y_test, y_pred), 4)
plt.plot(fpr,tpr,label="DecisionTreeClassifier, AUC="+str(auc))


#add legend
plt.legend()


# In[ ]:




