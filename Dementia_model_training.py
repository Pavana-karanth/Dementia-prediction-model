#!/usr/bin/env python
# coding: utf-8

# In[45]:


# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier

# Load your dataset
df = pd.read_csv(r"C:\Users\pavan\OneDrive\Desktop\dementia pred dataset\health_dementia_data.csv")

# Display basic information about the dataset
print(df.info())

# Display summary statistics
print("summary stats")
print(df.describe())

# Check for missing values
print("missing values checkk")
print(df.isnull().sum())

# Visualize the distribution of numerical features
print("visualise distribution of numerical features")
sns.set(style="whitegrid")
plt.figure(figsize=(10, 6))

# Example: Boxplot for numerical columns
sns.boxplot(data=df.select_dtypes(include=['float64', 'int64']))

# Customize plot if needed
plt.title('Boxplot of Numerical Features')
plt.xticks(rotation=45)
plt.show()


# In[46]:


correlation_matrix = df.corr()
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm")
plt.title("Correlation Matrix")
plt.show()


# In[47]:


X = df.drop("Dementia", axis=1)
y = df["Dementia"]

rf_model = RandomForestClassifier()
rf_model.fit(X, y)

feature_importances = pd.Series(rf_model.feature_importances_, index=X.columns)
feature_importances.sort_values(ascending=False).plot(kind='bar')
plt.title("Feature Importance from Random Forest")
plt.show()


# In[48]:


from sklearn.feature_selection import SelectKBest, chi2

# Assuming you have already defined X and y
X_new = SelectKBest(chi2, k='all').fit(X, y)

# Transform the features and store them in a DataFrame
X_selected = pd.DataFrame(X_new.transform(X), columns=X.columns[X_new.get_support()])

# Display scores and p-values
scores = pd.DataFrame({'Feature': X_selected.columns, 'Score': X_new.scores_, 'P-value': X_new.pvalues_})
print(scores)


# In[49]:


from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression()
rfe = RFE(logreg, n_features_to_select=1)
fit = rfe.fit(X, y)

ranking_df = pd.DataFrame({'Feature': X.columns, 'Ranking': rfe.ranking_})
print(ranking_df.sort_values(by='Ranking'))


# In[50]:


from sklearn.feature_selection import mutual_info_classif

mi_scores = mutual_info_classif(X, y)
mi_scores = pd.Series(mi_scores, index=X.columns)
mi_scores.sort_values(ascending=False).plot(kind='bar')
plt.title("Mutual Information Scores")
plt.show()


# In[51]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, random_state=42)


# In[52]:


from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Example with Logistic Regression
from sklearn.linear_model import LogisticRegression

# Create and train the model
model = LogisticRegression()
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1 Score: {f1:.2f}")


# In[53]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split

# Assuming you have already split your data into X_train, X_test, y_train, y_test
# If not, you can use train_test_split as mentioned in the previous responses.

# Create and train the Decision Tree model
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)

# Make predictions on the test set
y_pred_dt = dt_model.predict(X_test)

# Evaluate the Decision Tree model
accuracy_dt = accuracy_score(y_test, y_pred_dt)
precision_dt = precision_score(y_test, y_pred_dt)
recall_dt = recall_score(y_test, y_pred_dt)
f1_dt = f1_score(y_test, y_pred_dt)

print("Decision Tree Model:")
print(f"Accuracy: {accuracy_dt:.2f}")
print(f"Precision: {precision_dt:.2f}")
print(f"Recall: {recall_dt:.2f}")
print(f"F1 Score: {f1_dt:.2f}")


# In[54]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split

# Assuming you have already split your data into X_train, X_test, y_train, y_test
# If not, you can use train_test_split as mentioned in the previous responses.

# Create and train the Random Forest model
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)

# Make predictions on the test set
y_pred_rf = rf_model.predict(X_test)

# Evaluate the Random Forest model
accuracy_rf = accuracy_score(y_test, y_pred_rf)
precision_rf = precision_score(y_test, y_pred_rf)
recall_rf = recall_score(y_test, y_pred_rf)
f1_rf = f1_score(y_test, y_pred_rf)

print("Random Forest Model:")
print(f"Accuracy: {accuracy_rf:.2f}")
print(f"Precision: {precision_rf:.2f}")
print(f"Recall: {recall_rf:.2f}")
print(f"F1 Score: {f1_rf:.2f}")


# In[55]:


from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split

# Assuming you have already split your data into X_train, X_test, y_train, y_test
# If not, you can use train_test_split as mentioned in the previous responses.

# Create and train the SVM model
svm_model = SVC(random_state=42)
svm_model.fit(X_train, y_train)

# Make predictions on the test set
y_pred_svm = svm_model.predict(X_test)

# Evaluate the SVM model
accuracy_svm = accuracy_score(y_test, y_pred_svm)
precision_svm = precision_score(y_test, y_pred_svm)
recall_svm = recall_score(y_test, y_pred_svm)
f1_svm = f1_score(y_test, y_pred_svm)

print("Support Vector Machines (SVM) Model:")
print(f"Accuracy: {accuracy_svm:.2f}")
print(f"Precision: {precision_svm:.2f}")
print(f"Recall: {recall_svm:.2f}")
print(f"F1 Score: {f1_svm:.2f}")


# In[56]:


from sklearn.model_selection import cross_val_score, KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

# Assuming you have already defined X and y
rf_model = RandomForestClassifier(random_state=42)
dt_model = DecisionTreeClassifier(random_state=42)

kfold = KFold(n_splits=5, shuffle=True, random_state=42)

rf_scores = cross_val_score(rf_model, X, y, cv=kfold, scoring='f1')
print(f"Random Forest Cross-Validation F1 Score: {rf_scores.mean():.2f} (+/- {rf_scores.std():.2f})")

dt_scores = cross_val_score(dt_model, X, y, cv=kfold, scoring='f1')
print(f"Decision Tree Cross-Validation F1 Score: {dt_scores.mean():.2f} (+/- {dt_scores.std():.2f})")


# In[57]:


dt_param_grid = {
    'max_depth': [None, 5, 10, 15],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

from sklearn.model_selection import GridSearchCV

dt_grid_search = GridSearchCV(DecisionTreeClassifier(random_state=42),
                              param_grid=dt_param_grid,
                              scoring='f1',
                              cv=5,
                              verbose=1,
                              n_jobs=-1)

dt_grid_search.fit(X, y)

best_dt_params = dt_grid_search.best_params_
print("Best Decision Tree Hyperparameters:", best_dt_params)


# In[ ]:


rf_param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 5, 10, 15],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

rf_grid_search = GridSearchCV(RandomForestClassifier(random_state=42),
                              param_grid=rf_param_grid,
                              scoring='f1',
                              cv=5,
                              verbose=1,
                              n_jobs=-1)

rf_grid_search.fit(X, y)

best_rf_params = rf_grid_search.best_params_
print("Best Random Forest Hyperparameters:", best_rf_params)


# In[ ]:


best_dt_model = DecisionTreeClassifier(**best_dt_params, random_state=42)
best_dt_model.fit(X, y)

best_rf_model = RandomForestClassifier(**best_rf_params, random_state=42)
best_rf_model.fit(X, y)

# Print results for Decision Tree
dt_tuned_scores = cross_val_score(best_dt_model, X, y, cv=5, scoring='f1')
print(f"Tuned Decision Tree F1 Score: {dt_tuned_scores.mean():.2f} (+/- {dt_tuned_scores.std():.2f})")

# Print results for Random Forest
rf_tuned_scores = cross_val_score(best_rf_model, X, y, cv=5, scoring='f1')
print(f"Tuned Random Forest F1 Score: {rf_tuned_scores.mean():.2f} (+/- {rf_tuned_scores.std():.2f})")


# In[ ]:


dt_tuned_scores = cross_val_score(best_dt_model, X, y, cv=5, scoring='f1')
rf_tuned_scores = cross_val_score(best_rf_model, X, y, cv=5, scoring='f1')

print(f"Tuned Decision Tree F1 Score: {dt_tuned_scores.mean():.2f} (+/- {dt_tuned_scores.std():.2f})")
print(f"Tuned Random Forest F1 Score: {rf_tuned_scores.mean():.2f} (+/- {rf_tuned_scores.std():.2f})")


# In[ ]:


from joblib import dump

# Save the best models after hyperparameter tuning
dump(best_dt_model, 'best_decision_tree_model.joblib')



# In[ ]:


dump(best_rf_model, 'best_random_forest_model.joblib')


# In[ ]:


get_ipython().system('jupyter nbconvert --to script Dementia_model_training.ipynb')


# In[ ]:


import sklearn
print(sklearn.__version__)


# In[ ]:


from joblib import dump

dump(X, 'X.joblib')


# In[ ]:




