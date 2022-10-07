#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt  
import pandas as pd
import seaborn as sns
import matplotlib as mpl
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression,Lasso,Ridge
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from math import sqrt


# In[2]:


pd.options.display.max_columns=100


# In[3]:


df_store = pd.read_csv('stores.csv')


# In[4]:


df_train = pd.read_csv('train.csv')


# In[5]:


df_features = pd.read_csv('features.csv')


# In[6]:


df_store.head()


# In[7]:


df_train.head()


# In[8]:


df_features.head()


# In[9]:


# merging 3 different sets
df = df_train.merge(df_features, on=['Store', 'Date'], how='inner').merge(df_store, on=['Store'], how='inner')
df.head(5)


# In[10]:


df.drop(['IsHoliday_y'], axis=1,inplace=True)


# In[11]:


df.rename(columns={'IsHoliday_x':'IsHoliday'},inplace=True) 


# In[12]:


df.head()


# In[13]:


df.shape


# In[14]:


df['Store'].nunique()


# In[15]:


df['Dept'].nunique()


# In[16]:


store_dept_table = pd.pivot_table(df, index='Store', columns='Dept',
                                  values='Weekly_Sales', aggfunc=np.mean)
display(store_dept_table)


# In[17]:


df.loc[df['Weekly_Sales']<=0]


# In[18]:


df = df.loc[df['Weekly_Sales'] > 0]


# In[19]:


df.shape 


# In[20]:


df['Date'].head(5).append(df['Date'].tail(5))


# In[21]:


sns.barplot(x='IsHoliday', y='Weekly_Sales', data=df)


# In[22]:


df.isna().sum()


# In[23]:


df = df.fillna(0)


# In[24]:


df.isna().sum()


# In[25]:


df.describe()


# In[26]:


# Super bowl dates in train set
df.loc[(df['Date'] == '2010-02-12')|(df['Date'] == '2011-02-11')|(df['Date'] == '2012-02-10'),'Super_Bowl'] = True
df.loc[(df['Date'] != '2010-02-12')&(df['Date'] != '2011-02-11')&(df['Date'] != '2012-02-10'),'Super_Bowl'] = False


# In[27]:


# Labor day dates in train set
df.loc[(df['Date'] == '2010-09-10')|(df['Date'] == '2011-09-09')|(df['Date'] == '2012-09-07'),'Labor_Day'] = True
df.loc[(df['Date'] != '2010-09-10')&(df['Date'] != '2011-09-09')&(df['Date'] != '2012-09-07'),'Labor_Day'] = False


# In[28]:


# Thanksgiving dates in train set
df.loc[(df['Date'] == '2010-11-26')|(df['Date'] == '2011-11-25'),'Thanksgiving'] = True
df.loc[(df['Date'] != '2010-11-26')&(df['Date'] != '2011-11-25'),'Thanksgiving'] = False


# In[29]:


#Christmas dates in train set
df.loc[(df['Date'] == '2010-12-31')|(df['Date'] == '2011-12-30'),'Christmas'] = True
df.loc[(df['Date'] != '2010-12-31')&(df['Date'] != '2011-12-30'),'Christmas'] = False


# In[30]:


df.groupby(['Christmas','Type'])['Weekly_Sales'].mean()


# In[31]:


df.groupby(['Labor_Day','Type'])['Weekly_Sales'].mean()


# In[32]:


df.groupby(['Thanksgiving','Type'])['Weekly_Sales'].mean()


# In[33]:


df.groupby(['Super_Bowl','Type'])['Weekly_Sales'].mean()


# In[34]:


my_data = [48.88, 37.77 , 13.33 ]  #percentages
my_labels = 'Type A','Type B', 'Type C' # labels
plt.pie(my_data,labels=my_labels,autopct='%1.1f%%', textprops={'fontsize': 15}) #plot pie type and bigger the labels
plt.axis('equal')
mpl.rcParams.update({'font.size': 20}) #bigger percentage labels

plt.show()


# In[35]:


# Plotting avg wekkly sales according to holidays by types
plt.style.use('seaborn-poster')
labels = ['Thanksgiving', 'Super_Bowl', 'Labor_Day', 'Christmas']
A_means = [27397.77, 20612.75, 20004.26, 18310.16]
B_means = [18733.97, 12463.41, 12080.75, 11483.97]
C_means = [9696.56,10179.27,9893.45,8031.52]

x = np.arange(len(labels))  # the label locations
width = 0.25  # the width of the bars

fig, ax = plt.subplots(figsize=(16, 8))
rects1 = ax.bar(x - width, A_means, width, label='Type_A')
rects2 = ax.bar(x , B_means, width, label='Type_B')
rects3 = ax.bar(x + width, C_means, width, label='Type_C')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Weekly Avg Sales')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()


def autolabel(rects):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')


autolabel(rects1)
autolabel(rects2)
autolabel(rects3)

plt.axhline(y=17094.30,color='r') # holidays avg
plt.axhline(y=15952.82,color='green') # not-holiday avg

fig.tight_layout()

plt.show()


# In[36]:


df.sort_values(by='Weekly_Sales',ascending=False).head(5)


# In[37]:


df_store.groupby('Type').describe()['Size'].round(2)


# In[38]:


x = df['Dept']
y = df['Weekly_Sales']
plt.figure(figsize=(15,5))
plt.title('Weekly Sales by Department')
plt.xlabel('Departments')
plt.ylabel('Weekly Sales')
plt.scatter(x,y)
plt.show()


# In[39]:


plt.figure(figsize=(30,10))
fig = sns.barplot(x='Dept', y='Weekly_Sales', data=df)


# In[40]:


x = df['Store']
y = df['Weekly_Sales']
plt.figure(figsize=(15,5))
plt.title('Weekly Sales by Store')
plt.xlabel('Stores')
plt.ylabel('Weekly Sales')
plt.scatter(x,y)
plt.show()


# In[41]:


plt.figure(figsize=(20,6))
fig = sns.barplot(x='Store', y='Weekly_Sales', data=df)


# In[42]:


df["Date"] = pd.to_datetime(df["Date"]) # convert to datetime
df['week'] =df['Date'].dt.week
df['month'] =df['Date'].dt.month 
df['year'] =df['Date'].dt.year


# In[43]:


df.groupby('month')['Weekly_Sales'].mean()


# In[44]:


df.groupby('year')['Weekly_Sales'].mean()


# In[45]:


fig = sns.barplot(x='month', y='Weekly_Sales', data=df)


# In[46]:


df.groupby('week')['Weekly_Sales'].mean().sort_values(ascending=False).head()


# In[47]:


plt.figure(figsize=(20,6))
fig = sns.barplot(x='week', y='Weekly_Sales', data=df)


# In[48]:


fuel_price = pd.pivot_table(df, values = "Weekly_Sales", index= "Fuel_Price")
fuel_price.plot()


# In[49]:


temp = pd.pivot_table(df, values = "Weekly_Sales", index= "Temperature")
temp.plot()


# In[50]:


CPI = pd.pivot_table(df, values = "Weekly_Sales", index= "CPI")
CPI.plot()


# In[51]:


unemployment = pd.pivot_table(df, values = "Weekly_Sales", index= "Unemployment")
unemployment.plot()


# In[52]:


df['Date'] = pd.to_datetime(df['Date'])


# In[53]:


df_encoded = df.copy()


# In[54]:


type_group = {'A':1, 'B': 2, 'C': 3}  # changing A,B,C to 1-2-3
df_encoded['Type'] = df_encoded['Type'].replace(type_group)


# In[55]:


df_encoded['Super_Bowl'] = df_encoded['Super_Bowl'].astype(bool).astype(int)


# In[56]:


df_encoded['Thanksgiving'] = df_encoded['Thanksgiving'].astype(bool).astype(int)


# In[57]:


df_encoded['Labor_Day'] = df_encoded['Labor_Day'].astype(bool).astype(int)


# In[58]:


df_encoded['Christmas'] = df_encoded['Christmas'].astype(bool).astype(int)


# In[59]:


df_encoded['IsHoliday'] = df_encoded['IsHoliday'].astype(bool).astype(int)


# In[60]:


df_new = df_encoded.copy()


# In[61]:


df_new


# In[62]:


drop_col = ['Super_Bowl','Labor_Day','Thanksgiving','Christmas']
df_new.drop(drop_col, axis=1, inplace=True)


# In[63]:


plt.figure(figsize = (12,10))
sns.heatmap(df_new.corr().abs())    # To see the correlations
plt.show()


# In[64]:


drop_col = ['Temperature','MarkDown4','MarkDown5','CPI','Unemployment']
df_new.drop(drop_col, axis=1, inplace=True)


# In[65]:


plt.figure(figsize = (12,10))
sns.heatmap(df_new.corr().abs())    # To see the correlations without dropping columns
plt.show()


# In[85]:


df2=df_new[:150000]


# In[86]:


x = df2.drop(['Weekly_Sales','Date'],axis=1)
y = df2.Weekly_Sales


# In[87]:


df1 = pd.DataFrame(columns=["Model", "Accuracy for train","MSE for train","MAE for train",
                            "Accuracy for test","MSE for test","MAE for test"])


# In[88]:


x_train,x_test,y_train,y_test = train_test_split(x,y,test_size =0.4, random_state = 42)


# In[89]:


def pred_model(model,x_train,y_train,x_test,y_test):
    c = model()
    c.fit(x_train,y_train)
    x_pred = c.predict(x_train)
    y_pred = c.predict(x_test)
    
    print(c)
    
    print("For Training Data \n --------------------------------")
    print("MAE: ",mean_absolute_error(y_train, x_pred))
    print("MSE: ",mean_squared_error(y_train, x_pred))
    print("r2: ",r2_score(y_train, x_pred))
    print("RMSE: ",np.sqrt(mean_squared_error(y_train, x_pred)))

    print("")
    print("For Test Data \n --------------------------------")
    print("MAE: ",mean_absolute_error(y_test, y_pred))
    print("MSE: ",mean_squared_error(y_test, y_pred))
    print("r2: ",r2_score(y_test, y_pred))
    print("RMSE: ",np.sqrt(mean_squared_error(y_test,y_pred)))
    
   # print(f'MSE: {mean_squared_error(y_test,y_pred)}')
    #print(f'MAE: {mean_absolute_error(y_test,y_pred)}')
    #print(f'R2 : {r2_score(y_test,y_pred)}')
    
    
    print("Residual Analysis:")
    plt.figure(figsize = (20,5))
    plt.scatter(y_train,(y_train-x_pred),color = "red",label = 'Training Predictions')
    plt.scatter(y_test,(y_test-y_pred),color = "green",label = 'Testing Predictions')
    plt.legend()
    plt.show()
    
    re = {}
    re["Model"] = c
    re["Accuracy for train"] = 100*(r2_score(y_train, x_pred))
    re["MSE for train"] = mean_squared_error(y_test, y_pred)
    re["MAE for train"] = mean_absolute_error(y_test, y_pred)
    re["Accuracy for test"] = 100*(r2_score(y_test, y_pred))
    re["MSE for test"] = mean_squared_error(y_test,y_pred)
    re["MAE for test"] = mean_absolute_error(y_test,y_pred)
    
    return re


# In[90]:


l = (LinearRegression,RandomForestRegressor,DecisionTreeRegressor)

for i in l:
    re = pred_model(i, x_train,y_train,x_test,y_test)
    df1 = df1.append(re, ignore_index = True)


# In[91]:


n1 = df1.Model.values
n1[1]='RandomForest()'


# In[92]:


print('Results for 80:20 Ratio : ')
df1


# In[93]:


import matplotlib.pyplot as plt
model = ['LinearRegression','RandomForestRegressor','DecisionTreeRegressor']
acc = df1['Accuracy for train'].tolist()
plt.bar(model,acc)
plt.show()


# In[100]:


n_estimators = [5,20,50,100] 
max_features = ['auto', 'sqrt'] 
max_depth = [int(x) for x in np.linspace(10, 120, num = 12)] 
min_samples_split = [2, 6, 10] 
min_samples_leaf = [1, 3, 4]
bootstrap = [True, False]

random_grid = {'n_estimators': n_estimators,

'max_features': max_features,

'max_depth': max_depth,

'min_samples_split': min_samples_split,

'min_samples_leaf': min_samples_leaf,

'bootstrap': bootstrap}


# In[101]:


rf = RandomForestRegressor()


# In[102]:


from sklearn.model_selection import RandomizedSearchCV
rf_random = RandomizedSearchCV(estimator = rf,param_distributions = random_grid,
               n_iter = 100, cv = 5, verbose=2, random_state=35, n_jobs = -1)
rf_random.fit(x_train, y_train)


# In[103]:


print ('Random grid: ', random_grid, '\n')
print ('Best Parameters: ', rf_random.best_params_, ' \n')


# In[105]:


import time
start = time.time()
randmf = RandomForestRegressor(n_estimators = 100, min_samples_split = 2,min_samples_leaf= 1, max_features = 'auto', 
                               max_depth=90,bootstrap=True)
randmf.fit( x_train, y_train) 

x_pred = randmf.predict(x_train)
y_pred = randmf.predict(x_test)
end = time.time()
m = end - start
print(f"Runtime of the program is {end - start}")


# In[ ]:





# In[106]:


print("For Training Data \n --------------------------------")
print("MAE: ",mean_absolute_error(y_train, x_pred))
print("MSE: ",mean_squared_error(y_train, x_pred))
print("r2: ",r2_score(y_train, x_pred))
print("RMSE: ",np.sqrt(mean_squared_error(y_train, x_pred)))

print("")
print("For Test Data \n --------------------------------")
print("MAE: ",mean_absolute_error(y_test, y_pred))
print("MSE: ",mean_squared_error(y_test, y_pred))
print("r2: ",r2_score(y_test, y_pred))
print("RMSE: ",np.sqrt(mean_squared_error(y_test,y_pred)))
    
print("Residual Analysis:")
plt.figure(figsize = (20,5))
plt.scatter(y_train,(y_train-x_pred),color = "red",label = 'Training Predictions')
plt.scatter(y_test,(y_test-y_pred),color = "green",label = 'Testing Predictions')
plt.legend()
plt.show()


# In[107]:


import time
start1 = time.time()
randmf1 = RandomForestRegressor() 
randmf1.fit( x_train, y_train) 

x_pred = randmf1.predict(x_train)
y_pred = randmf1.predict(x_test)
end1 = time.time()
n = end1 - start1
print(f"Runtime of the program is {end1 - start1}")


# In[108]:


print("For Training Data \n --------------------------------")
print("MAE: ",mean_absolute_error(y_train, x_pred))
print("MSE: ",mean_squared_error(y_train, x_pred))
print("r2: ",r2_score(y_train, x_pred))
print("RMSE: ",np.sqrt(mean_squared_error(y_train, x_pred)))

print("")
print("For Test Data \n --------------------------------")
print("MAE: ",mean_absolute_error(y_test, y_pred))
print("MSE: ",mean_squared_error(y_test, y_pred))
print("r2: ",r2_score(y_test, y_pred))
print("RMSE: ",np.sqrt(mean_squared_error(y_test,y_pred)))
    
print("Residual Analysis:")
plt.figure(figsize = (20,5))
plt.scatter(y_train,(y_train-x_pred),color = "red",label = 'Training Predictions')
plt.scatter(y_test,(y_test-y_pred),color = "green",label = 'Testing Predictions')
plt.legend()
plt.show()


# In[109]:


model = ['With tuning','Without tuning']
acc = [m,n]
plt.bar(model,acc)
plt.show()


# In[ ]:




