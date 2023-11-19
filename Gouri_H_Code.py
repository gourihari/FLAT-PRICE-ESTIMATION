#!/usr/bin/env python
# coding: utf-8

# **FLAT PRICE ESTIMATION**

# In[1]:


import pandas as pd
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
get_ipython().run_line_magic('matplotlib', 'inline')


# **ANALYSING THE TRAINING AND TEST DATA**

# In[2]:


train = pd.read_csv("training_data.csv")
target_train=pd.read_csv("training_data_targets.csv")
test=pd.read_csv("test_data.csv")


# In[3]:


test


# In[4]:


train.head()


# In[5]:


target_train.head()


# In[6]:


test.head()


# In[7]:


train=train.join(target_train)


# In[8]:


train.describe()


# In[9]:


num_rows = train.shape[0]

print(f"The train DataFrame has {num_rows} rows.")


# In[10]:


train.info()


# In[11]:


train.isna().sum()


# In[12]:


train['CITY'] = train['ADDRESS'].str.split(',').str.get(1)
train['CITY'].value_counts()


# In[13]:


train.head()


# In[14]:


sns.barplot(x='CITY', y='PRICE', data=train)
plt.title('Target Price by CITY')
plt.show()


# In[15]:


train.head()


# In[16]:


# Assuming 'Address' column has been extracted to 'City'
train.drop('ADDRESS', axis=1, inplace=True)


# In[17]:


train.head()


# **ONE-HOT ENCODING (Over Training Data)**

# In[18]:


import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import dask.dataframe as dd


# Identify categorical columns
#categorical_cols = train.select_dtypes(include=['object']).columns
categorical_cols=['BHK_NO.']

# Specify the number of chunks
num_chunks = 10

# Convert to Dask DataFrame
ddf = dd.from_pandas(train, npartitions=num_chunks)

# Apply one-hot encoding to each chunk
encoded_chunks = []
encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

for chunk in ddf.to_delayed():
    chunk_df = pd.DataFrame(chunk.compute())
    X_categorical = chunk_df[categorical_cols]
    X_encoded = encoder.fit_transform(X_categorical)
    encoded_chunk = pd.DataFrame(X_encoded, columns=encoder.get_feature_names_out(categorical_cols))
    encoded_chunks.append(encoded_chunk)

# Concatenate the encoded chunks
X_processed = pd.concat(encoded_chunks, axis=0)
# Display the processed features
#print(X_processed)
X_processed.reset_index(drop=True, inplace=True)

# Concatenate X_processed to the original train DataFrame
train_encoded = pd.concat([train, X_processed], axis=1)

# Display the updated DataFrame

X_processed=train_encoded.drop('BHK_NO.',axis=1)
print(X_processed)


# Converting NaN values

# In[19]:


X_combined_imputed = X_processed.fillna(0)
X_combined_imputed


# In[20]:


X_combined_imputed.drop('CITY',axis=1,inplace=True)
print(X_combined_imputed)


# In[ ]:


#the label 'CITY' is dropped before regression as there were some unseen labels.


# **REGRESSION USING VARIOUS MODELS**

# In[21]:


from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, r2_score

y = train['PRICE']
X_train, X_test, y_train, y_test = train_test_split(X_combined_imputed, y, test_size=0.2, random_state=42)

models = {
    'Linear Regression': LinearRegression(),
    'Decision Tree Regression': DecisionTreeRegressor(random_state=42),
    'Random Forest Regression': RandomForestRegressor(random_state=42),
    'Support Vector Regression': SVR()
}

# Training and evaluating each model
for model_name, model in models.items():
    # Training the model
    model.fit(X_train, y_train)

    # Making predictions
    y_pred = model.predict(X_test)

    # Evaluating the model
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f'{model_name} - Mean Absolute Error: {mae}')
    print(f'{model_name} - R2 Score: {r2}')
    print('\n')


# In[ ]:


#While comparing the R2 scores and mean absolute error, we find that Random Forest and Decision Tree has promising values.


# **GRID SEARCH ON RANDOM FOREST REGRESSION**

# In[23]:


# Define the model
model = RandomForestRegressor()

# Define the hyperparameter grid
param_grid = {
    'n_estimators': [75,100,120],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2,3,4]
}

# Create the grid search object with cross-validation
grid_search = GridSearchCV(model, param_grid, scoring='r2', n_jobs=-1)

# Perform grid search on the entire training set
grid_search.fit(X_train, y_train)

# Get the best hyperparameters
best_params = grid_search.best_params_

# Train the final model with the best hyperparameters on the entire training set
final_model = RandomForestRegressor(**best_params)
final_model.fit(X_train, y_train)

rf_preds = final_model.predict(X_test)
test_score=r2_score(y_test, rf_preds)


# In[24]:


print("Best hyperparameters:", best_params)
print("Test score:", test_score)


# In[25]:


rf_preds


# **RANDOM FOREST REGRESSION ON TEST DATA**

# In[26]:


test=pd.read_csv("test_data.csv")
test['CITY'] = test['ADDRESS'].str.split(',').str.get(1)


# In[27]:


test.drop('ADDRESS',axis=1,inplace=True)
print(test)


# **ONE HOT ENCODING**

# In[28]:


import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import dask.dataframe as dd


# Identify categorical columns
#categorical_cols = train.select_dtypes(include=['object']).columns
categorical_cols_test=['BHK_NO.']

# Specify the number of chunks
num_chunks = 10

# Convert to Dask DataFrame
ddf = dd.from_pandas(test, npartitions=num_chunks)

# Apply one-hot encoding to each chunk
encoded_chunks_test = []
encoder_test = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

for chunk in ddf.to_delayed():
    chunk_df_test = pd.DataFrame(chunk.compute())
    X_categorical_test = chunk_df_test[categorical_cols_test]
    X_encoded_test = encoder_test.fit_transform(X_categorical_test)
    encoded_chunk_test = pd.DataFrame(X_encoded_test, columns=encoder_test.get_feature_names_out(categorical_cols_test))
    encoded_chunks_test.append(encoded_chunk_test)

# Concatenate the encoded chunks
X_processed_test = pd.concat(encoded_chunks_test, axis=0)
# Display the processed features
#print(X_processed)
X_processed_test.reset_index(drop=True, inplace=True)

# Concatenate X_processed to the original train DataFrame
test_encoded = pd.concat([test, X_processed_test], axis=1)

# Display the updated DataFrame

X_processed_test=test_encoded.drop('BHK_NO.',axis=1)
print(X_processed_test)


# In[29]:


X_combined_imputed_test = X_processed_test.fillna(0)
X_combined_imputed_test


# In[30]:


X_combined_imputed_test.drop('CITY',axis=1,inplace=True)
# Display the updated DataFrame
print(X_combined_imputed_test)


# In[31]:


missing_cols_X_combined_imputed = set(X_combined_imputed.columns) - set(X_combined_imputed_test.columns)

# Add missing columns to df2 with zero values
for col in missing_cols_X_combined_imputed:
    X_combined_imputed_test[col] = 0
    
X_combined_imputed_test = X_combined_imputed_test[X_combined_imputed.columns]


# In[32]:


X_combined_imputed_test


# In[33]:


preds_final=final_model.predict(X_combined_imputed_test)


# In[34]:


print(preds_final)


# In[35]:


# Specify the path where you want to save the text file
output_file_path = 'predictedvalues_final.txt'

# Open the file in write mode
with open(output_file_path, 'w') as file:
    # Write each predicted value to a new line
    for prediction in preds_final:
        file.write(f'{prediction}\n')

print(f'Predicted values saved to {output_file_path}')

