#!/usr/bin/env python
# coding: utf-8

# **FLAT PRICE ESTIMATION**

# **Analysing The Training and Test Data**

# In[1]:


pip install pandas


# In[2]:


# Importing the libraries 
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
get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


train = pd.read_csv("training_data.csv")
target_train=pd.read_csv("training_data_targets.csv")
test=pd.read_csv("test_data.csv")


# In[4]:


train.head()


# In[5]:


train.drop(['ADDRESS'], axis =1)


# In[6]:


target_train.head()


# In[7]:


test.head()


# In[8]:


train.join(target_train)


# In[9]:


train_df = train.join(target_train)


# In[10]:


train_df.describe()


# In[11]:


num_rows = train_df.shape[0]

print(f"The train DataFrame has {num_rows} rows.")


# In[12]:


train_df.info()


# In[13]:


train_df.isna().sum()


# **DISTRIBUTION**

# In[14]:


numerical_columns = train_df.select_dtypes(include=['number'])

numerical_columns.columns


# In[15]:


train_df.shape


# In[16]:


train_df.dtypes


# In[17]:


train_df.nunique()


# In[18]:


train_df = train_df.drop(['ADDRESS'], axis =1)


# In[19]:


train_df.describe()


# **PLOTS**

# **Correlation Matrix**

# In[20]:


corr = train_df.corr()
corr.shape


# In[21]:


plt.figure(figsize=(20,20))
sns.heatmap(corr, cbar=True, square= True, fmt='.1f', annot=True, annot_kws={'size':15}, cmap='Reds')


# In[22]:


numerical_columns = train_df.select_dtypes(include=['number'])

numerical_columns.columns


# **Histogram**

# In[23]:


train.hist(figsize=(15,8))


# In[24]:


numerical_useful = ['UNDER_CONSTRUCTION', 'RERA', 'BHK_NO.', 'READY_TO_MOVE',
       'RESALE', 'PRICE', 'SQUARE_FT']
    
    
for col in numerical_useful:
    d_type = train_df[col].dtype
    
    if d_type != "object" and (train_df[col] > 1).any().any():
        plt.figure(figsize=(8, 4))
        sns.histplot(train_df[col], bins=50, kde=True)
        plt.title(f"Distribution of {col}")
        plt.xlabel(f'{col}')
        plt.show()
    
    counts = train_df[col].value_counts()  
    print(f"Counts for {col}:")
    print(counts)
    print("============================================================")


# In[25]:


train_df['LOG_PRICE']=np.log(train_df['PRICE'])
train_df['LOG_SQFT']=np.log(train_df['SQUARE_FT'])
log_columns = ['LOG_SQFT', 'LOG_PRICE' ]
train_df.head()


# In[26]:


plt.figure(figsize=(10, 6))

# Create a bar chart to visualize the distribution of 'price'
plt.subplot(1, 2, 1)
sns.histplot(data=train_df, x='LOG_PRICE', kde=True, color='skyblue')
plt.xlabel('LOG_PRICE')
plt.title('LOG_PRICE Distribution')

# Create a box plot to visualize the summary statistics of 'price'
plt.subplot(1, 2, 2)
sns.boxplot(data=train_df, y='LOG_PRICE', color='salmon')
plt.ylabel('LOG_PRICE')
plt.title('LOG_PRICE outliers')

# Adjust layout
plt.tight_layout()

# Show the plots
plt.show()


# In[27]:


# Create a bar chart and data distribution plot using Seaborn
plt.figure(figsize=(10, 6))

# Create a bar chart to visualize the distribution of 'logofSQFT'
plt.subplot(1, 2, 1)
sns.histplot(data=train_df, x='LOG_SQFT', kde=True, color='skyblue')
plt.xlabel('LOG_SQFT')
plt.title('LOG_SQFT Distribution')

# Create a box plot to visualize the summary statistics of 'logofSQFT'
plt.subplot(1, 2, 2)
sns.boxplot(data=train_df, y='LOG_SQFT', color='salmon')
plt.ylabel('LOG_SQFT')
plt.title('LOG_SQFT outliers')

# Adjust layout
plt.tight_layout()

# Show the plots
plt.show()


# In[28]:


sns.scatterplot(x='SQUARE_FT', y='PRICE', data=train_df)
plt.title('Target Price by SQUARE_FT')
plt.show()


# In[29]:


#Dropping irrelevant variables before performing various models


# In[30]:


train_df.drop([ 'LOG_PRICE', 'LOG_SQFT'], axis=1, inplace=True)


# In[31]:


train_df


# **REGRESSION MODELS**

# **LINEAR REGRESSION**

# In[32]:


train_df.head()


# In[33]:


training_data = train_df.drop(["PRICE"],axis=1)
target = train_df["PRICE"]


# In[34]:


X_train, X_test, y_train, y_test = train_test_split(training_data, target, test_size=0.2, random_state=22)


# In[35]:


model = LinearRegression()
model.fit(X_train,y_train)
print("Linear regression R2 score:")
model.score(X_train, y_train)


# In[36]:


model = LinearRegression()
model.fit(X_train, y_train)
preds = model.predict(X_test)


# In[37]:


mae = mean_absolute_error(y_test, preds)
print("Mean Absolute Error:", mae)
mse = mean_squared_error(y_test, preds)
print("Mean Squared Error:", mse)


#  The R-squared (R2) score measures the goodness of fit of a regression model. It indicates the proportion of the variance in the dependent variable (the target) that is predictable from the independent variables (the features) in the model.
# An R2 score of 0.158 suggests that approximately 15.8% of the variance in the target variable can be explained by the features included in the linear regression model. In other words: 15.8% of the variability in the target variable is accounted for by the linear relationship with the features.
# The remaining variability, around 84.2%, is unexplained by the model.An R2 score closer to 1.0 indicates that a higher proportion of the variance in the target variable is explained by the model, suggesting a better fit. Conversely, a score closer to 0 suggests that the model does not explain much of the variance in the target variable.
# It's crucial to interpret this score in context. While 0.158 might seem low, the significance can vary depending on the domain, the nature of the data, and the problem being addressed. Further improvement or exploration of additional features might enhance the model's predictive power. Also, the mae and mse values should be low for better predictive performance and here, the values are a bit higher.
# Now, we will check with decision tree regression.

# **GRID SEARCH ON LINEAR REGRESSION**

# In[38]:


X_train, X_test, y_train, y_test = train_test_split(training_data, target, test_size=0.2, random_state=42)

# Define the model
model = LinearRegression()

# Define the hyperparameter grid
param_grid = {
    'fit_intercept': [True, False],
}

# Create the grid search object with cross-validation
grid_search = GridSearchCV(model, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)

# Perform grid search on the entire training set
grid_search.fit(X_train, y_train)

# Get the best hyperparameters
best_params = grid_search.best_params_

# Train the final model with the best hyperparameters on the entire training set
final_model = LinearRegression(
    fit_intercept=best_params['fit_intercept']
)
final_model.fit(X_train, y_train)

# Evaluate the final model on the test set
y_pred = final_model.predict(X_test)
test_score = mean_squared_error(y_test, y_pred)

# Evaluate the final model on the test set
y_pred = final_model.predict(X_test)

# Calculate the R-squared score
r2 = r2_score(y_test, y_pred)

# Print the R-squared score
print(f'R-squared Score: {r2}')
print(f'Best Hyperparameters: {best_params}')


# **DECISION TREE REGRESSOR**

# In[39]:


model = DecisionTreeRegressor(max_depth=10)
model.fit(X_train,y_train)
preds=model.predict(X_test)
print("Decision tree R2 score:")
r2_score(preds,y_test)


# An R-squared (R2) score of approximately 0.88 in the context of a decision tree model suggests that around 88.0% of the variance in the target variable is explained by the features utilized in the model.
# Interpreting this score: 88.0% Explanation: The decision tree model explains a substantial portion of the variability in the target variable using the provided features.#12.0% Unexplained Variance: Around 12.0% of the variance in the target variable remains unexplained by the model.An R2 score close to 1.0 indicates a model that fits the data well, explaining a higher proportion of the variance. It's a strong indicator that the features used in the decision tree model are capturing a significant amount of the variability present in the target variable. This suggests a good fit, but as always, understanding the context of the data and the problem domain is important for a comprehensive evaluation

#  Mean absolute error and mean squared error of decision tree

# In[40]:


print("Decision tree mean absolute error:")
mae = mean_absolute_error(y_test, preds)
mae


# In[41]:


print("Decision tree mean squared error:")
mse = mean_squared_error(y_test, preds)
mse


# **GRID SEARCH ON DECISION TREE REGRESSOR**

# In[42]:


X_train, X_test, y_train, y_test = train_test_split(training_data, target, test_size=0.2, random_state=42)

# Define the model
model = DecisionTreeRegressor()

# Define the hyperparameter grid
param_grid = {
    'max_depth': [None,5, 10],
    'min_samples_split': [10,20,25,30],
    'min_samples_leaf': [5,6,7]
}

# Create the grid search object with cross-validation
grid_search = GridSearchCV(model, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)

# Perform grid search on the entire training set
grid_search.fit(X_train, y_train)

# Get the best hyperparameters
best_params = grid_search.best_params_

# Train the final model with the best hyperparameters on the entire training set
final_model = DecisionTreeRegressor(
    max_depth=best_params['max_depth'],
    min_samples_split=best_params['min_samples_split'],
    min_samples_leaf=best_params['min_samples_leaf']
)
final_model.fit(X_train, y_train)

# Evaluate the final model on the test set
y_pred = final_model.predict(X_test)

# Calculate the metrics on the test set
test_mse = mean_squared_error(y_test, y_pred)
test_r2 = r2_score(y_test, y_pred)

# Print the best hyperparameters and test metrics
print(f'Best Hyperparameters: {best_params}')
print(f'Mean Squared Error on Test Set: {test_mse}')
print(f'R2 Score on Test Set: {test_r2}')


# From the above two models, comparing the values of mae and mse,The Decision Tree model outperforms the Linear Regression model in both metrics. It demonstrates lower errors in terms of both the absolute and squared differences between predicted and actual flat prices.Considering both MAE and MSE together, the Decision Tree model consistently shows better performance in predicting flat prices compared to the Linear Regression model for your specific dataset and problem. Therefore, the Decision Tree model seems to be the better choice for this flat price estimation task based on these metrics.

# **RANDOM FOREST REGRESSION**

# In[44]:


rf_model = RandomForestRegressor(n_estimators=100, random_state=42)  # You can adjust n_estimators as needed
rf_model.fit(X_train, y_train)
rf_preds = rf_model.predict(X_test)


# In[45]:


rf_mae = mean_absolute_error(y_test, rf_preds)
rf_mse = mean_squared_error(y_test, rf_preds)
r2 = r2_score(y_test, rf_preds)

print("Random Forest Regression Mean Absolute Error:", rf_mae)
print("Random Forest Regression Mean Squared Error:", rf_mse)
print(f'R2 Score on Test Set: {r2}')


# **GRID SEARCH ON RANDOM TREE REGRESSOR**

# In[46]:


X_train, X_test, y_train, y_test = train_test_split(training_data, target, test_size=0.2, random_state=42)

# Define the model
model = RandomForestRegressor()

# Define the hyperparameter grid
param_grid = {
    'n_estimators': [75,100,120],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2,3,4]
}

# Create the grid search object with cross-validation
grid_search = GridSearchCV(model, param_grid, cv=5, scoring='r2', n_jobs=-1)

# Perform grid search on the entire training set
grid_search.fit(X_train, y_train)

# Get the best hyperparameters
best_params = grid_search.best_params_

# Train the final model with the best hyperparameters on the entire training set
final_model = RandomForestRegressor(**best_params)
final_model.fit(X_train, y_train)

rf_preds = rf_model.predict(X_test)
test_score=r2_score(y_test, rf_preds)


# In[47]:


print("Best hyperparameters:", best_params)
print("Test score:", test_score)


# Random Forest model exhibits significantly lower errors in both Mean Absolute Error (MAE) and Mean Squared Error (MSE) compared to the earlier Linear Regression and Decision Tree models.

# **SUPPORT VECTOR REGRESSION**

# In[48]:


# Feature scaling (important for SVR)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# In[49]:


# Initialize and train a Support Vector Regression model
svr_model = SVR(kernel='rbf')  
svr_model.fit(X_train_scaled, y_train)


# In[50]:


svr_preds_scaled = svr_model.predict(X_test_scaled)

# Evaluate the SVR model on the scaled test set
mse_svr_scaled = mean_squared_error(y_test, svr_preds_scaled)
r2_svr_scaled = r2_score(y_test, svr_preds_scaled)

print(f'Mean Squared Error with SVR on Scaled Test Set: {mse_svr_scaled}')
print(f'R2 Score with SVR on Scaled Test Set: {r2_svr_scaled}')


# In[51]:


svr_preds = svr_model.predict(X_test_scaled)


# In[52]:


# Calculate Mean Absolute Error (MAE) and Mean Squared Error (MSE)
svr_mae = mean_absolute_error(y_test, svr_preds)
svr_mse = mean_squared_error(y_test, svr_preds)

print("SVR Mean Absolute Error:", svr_mae)
print("SVR Mean Squared Error:", svr_mse)


# **Comparison**

# In[53]:


# Initialize models
linear_reg_model = LinearRegression()
decision_tree_model = DecisionTreeRegressor(random_state=42)
random_forest_model = RandomForestRegressor(n_estimators=100, random_state=42)
svr_model = SVR(kernel='rbf')

results = []


def evaluate_model(model, model_name, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    
    results.append({'Model': model_name, 'R2 Score': r2, 'MAE': mae, 'MSE': mse})

# Evaluate each model
evaluate_model(linear_reg_model, 'Linear Regression', X_train, X_test, y_train, y_test)
evaluate_model(decision_tree_model, 'Decision Tree Regression', X_train, X_test, y_train, y_test)
evaluate_model(random_forest_model, 'Random Forest Regression', X_train, X_test, y_train, y_test)
evaluate_model(svr_model, 'Support Vector Regression', X_train, X_test, y_train, y_test)


results_df = pd.DataFrame(results)


print(results_df)


# Here, while comparing we get that random forest regression has the least errors while compared to all the other models. Hence, we shall use this model for predicting flat prices.

# **RANDOM FOREST REGRESSION PREDICTED PRICES USING TEST DATA**

# In[54]:


print(test.columns)


# In[56]:


test = test.drop('ADDRESS', axis=1)


# In[57]:


# Use the best hyperparameters obtained from grid search
best_n_estimators = best_params['n_estimators']
best_max_depth = best_params['max_depth']
best_min_samples_split = best_params['min_samples_split']

# Create a new instance of the RandomForestRegressor with the best hyperparameters
best_model = RandomForestRegressor(
    n_estimators=best_n_estimators,
    max_depth=best_max_depth,
    min_samples_split=best_min_samples_split
)


best_model.fit(X_train, y_train)


test_predictions = best_model.predict(test)


print(test_predictions)


# In[58]:


# Specify the path where you want to save the text file
output_file_path = 'predictedvalues.txt'

# Open the file in write mode
with open(output_file_path, 'w') as file:
    # Write each predicted value to a new line
    for prediction in test_predictions:
        file.write(f'{prediction}\n')

print(f'Predicted values saved to {output_file_path}')

