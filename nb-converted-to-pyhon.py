# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from IPython import get_ipython

# %%

import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid') 
import pandas as pd 
import numpy as np
import warnings
from sklearn import linear_model
#import seaborn as sns
#from sklearn.datasets import load_iris
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")
get_ipython().run_line_magic('matplotlib', 'inline')


# %%
source_data_file = pd.read_csv(r'raw-data\all-ops-2006-2013-extract-2.csv', parse_dates=['Date'], thousands=',', decimal='.')
#.dropna()

source_columns_list_initial = ['Executed','Date','Expense amount','Income amount']
#['Executed','Date','Income account','Income amount','Income balance','Expense account','Expense amount','Expense balance','Comment','Category','Agent']

source_columns_list_to_delete = ['Executed']

training_data_percentage = 0.75


# %%
#######################################
# vars setup/reset
# cleaning the variable
if 'my_data_frame' in locals():
  del my_data_frame  
if 'source_columns_list' in locals():
  del source_columns_list

source_columns_list = list (set(source_columns_list_initial) - set (source_columns_list_to_delete))  
print ('source_columns_list',source_columns_list)
#######################################


# %%
#######################################
## Cleaning data to process ###########
# loading initial DataFrame 
my_data_frame = pd.DataFrame(source_data_file, columns= source_columns_list_initial ).replace(r'Света',r'Agent-1').replace(r'Вадим',r'Agent-2').replace(r'Архипп',r'Agent-3')
# .drop_duplicates(inplace = True) #.fillna('')
my_data_frame.info()

# dropping records with Executed != +my_data_frame.drop(my_data_frame[my_data_frame.Executed != '+'].index, inplace=True) 
my_data_frame.drop(columns = source_columns_list_to_delete , inplace=True) 
# exclude 2013 
my_data_frame.drop( my_data_frame[my_data_frame['Income amount'] >  150000 ].index, inplace=True) 
my_data_frame.drop( my_data_frame[my_data_frame['Expense amount'] <  -150000 ].index, inplace=True) 
my_data_frame.sort_values(by=['Date'])
#my_data_frame.drop( my_data_frame[my_data_frame['Date'] <  ???? ].index, inplace=True) 

#########################################
# lets visualise
        # my_data_frame.reindex(columns=['Income amount', 'Expense amount','Expense balance'])
        ## sns.pairplot(my_data_frame, x_vars=['Income amount', 'Expense amount','Expense balance'],y_vars='Date', size=4, aspect=1, kind='scatter')
        ## plt.show()
#########################################

#my_data_frame_size = int(my_data_frame.size / len(source_columns_list ))

#print ('DataFrame size = ' , my_data_frame_size)
#my_training_data_size = round (my_data_frame_size /100 * training_data_percentage ) 
#my_test_data_size = my_data_frame_size - my_training_data_size


#my_training_data = my_data_frame[0:my_training_data_size]
#my_training_data.head(my_training_data_size * len(source_columns_list ))
#my_test_data = my_data_frame[my_training_data_size:my_data_frame_size ]

#print ('my_training_data_size  = ' , my_training_data_size)
#print ('my_training_data.size  = ' , my_training_data.size)
#print ('my_test_data_size size = ' , my_test_data_size)
#print (my_training_data['Expense amount'].tail(3))
#print (my_test_data['Expense amount'].head(3))


# %%
my_grouped_dataset = my_data_frame.groupby(pd.Grouper(key='Date',freq="M")).sum().sort_values(by=['Date'], ascending=True).copy()
############# Negative or positive???
# my_grouped_dataset['Expense amount'] = my_grouped_dataset['Expense amount'].abs()

#my_grouped_dataset.reset_index()
#################################
###  Lagged Expenses for further analysis 
#################################
#my_grouped_dataset['Expense amount lagged'] = my_grouped_dataset['Expense amount'].shift(1)

my_grouped_dataset_training, my_grouped_dataset_testing = train_test_split ( my_grouped_dataset, test_size=0.3)

## checking columns on the Dataframe
#col_mapping = [f"{c[0]}:{c[1]}" for c in enumerate(my_grouped_dataset_testing.columns)]
#print(col_mapping) # ['0:Expense amount', '1:Income amount']
#print(my_grouped_dataset_training['Expense amount'])


print(my_grouped_dataset_training.head(10))


# %%
plt.clf()
my_grouped_dataset.sort_values(by=['Date']).plot(kind='bar',figsize = (20,5)) 
plt.xlabel("Months")
plt.ylabel("Expenses/Income")

plt.show()


# %%
X_train = my_grouped_dataset_training[['Income amount']] 
Y_train = my_grouped_dataset_training[['Expense amount']] 



regr_model = linear_model.LinearRegression()
regr_model.fit(X_train, Y_train)

print('Intercept: \n', regr_model.intercept_)
print('Coefficients: \n', regr_model.coef_)

# print('regr_model.predict(236606.02) = ',regr_model.predict([[515269.32]]))


# %%

#my_grouped_dataset_testing['Predicted Expense'] = my_grouped_dataset_testing.apply( lambda row: str(regr_model.predict([[ row['Income amount'] ]])) + ' for '+str(row['Income amount']) , axis = 1)

my_grouped_dataset_testing['Predicted Expense'] = my_grouped_dataset_testing.apply( lambda row: float(regr_model.predict([[ row['Income amount'] ]]))  , axis = 1)

#print(my_grouped_dataset_testing['Predicted Expense'].head (10))
print(my_grouped_dataset_testing.head (10))
#print (df_out)
    


# %%
plt.clf()
my_grouped_dataset_testing.sort_values(by=['Date']).plot(kind='bar',figsize = (20,5)) 
plt.xlabel("Months")
plt.ylabel("Expenses/Income")

plt.show()


# %%
# Compute the root-mean-square

rms = np.sqrt(mean_squared_error(my_grouped_dataset_testing[['Expense amount']] , my_grouped_dataset_testing[['Predicted Expense']]))
print(rms)
# 171995.66316816062


# %%



