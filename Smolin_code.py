#!/usr/bin/env python
# coding: utf-8

# explore the data, build a predictive model to predict the personal_loan_purpose_code of customers.
#
# Authors: Nikolai Smolin
# 
# Loading python libraries which we will use in this project

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn import model_selection
import lightgbm
from sklearn import metrics
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn_pandas import CategoricalImputer, DataFrameMapper


# Functions:

def explore_raw_data(df):
    """Function to explore raw data.
    ----- Args:
        data: pandas.DataFrame
    ----- Returns:
        pandas.DataFrame
    """  
    # number of unique values
    df1 = pd.DataFrame(df.apply(pd.Series.nunique)) 
    df1.columns = ['Unique Values']
    
    # number of NaNs that are present
    df2 = pd.DataFrame(df.isnull().sum()) 
    df2.columns = ['Number of NaNs']
    
    # data types for each variable
    df3 = pd.DataFrame(df.dtypes) 
    df3.columns = ['DataType']
    
    # gives basic statistics for numericals
    df4 = df.describe().T 
    finalDf = pd.concat([df1, df2, df3, df4], axis=1)
    return finalDf


# Loading data for project
data_test = pd.read_csv('data_test.csv', index_col=0)
print("Shape of the test data set: ", data_test.shape)
print("Number of duplicate rows: ", len(data_test)-len(data_test.drop_duplicates()))

data_training = pd.read_csv('data_training.csv', index_col=0)
print("Shape of the training data set: ",data_training.shape)
print("Number of duplicate rows: ", len(data_training)-len(data_training.drop_duplicates()))


# Training set has 959922 duplicate rows
# we will remove it
data_training = data_training.drop_duplicates()
print("Shape of the training data set after droping duplicates: ", data_training.shape)


# Exploring training data set
explore_raw_data(data_training)

data_training.head()

# Exploring test data set
explore_raw_data(data_test)


data_test.head()


# Counting number of different reasond for personal loan
# We can see that data set in imbalance meaning that some PL purposes dominating
# training data set. This is an indication that we need to resample data set for model training
#sns.countplot(x = 'personal_loan_purpose_code', data = data_training)
#plt.xticks(rotation=90)
#plt.title('Count Plot of PL purpose')
#plt.show()



# dropping "form_id" column: this column is unique user identifier 
# it will be not used for training or scoring
data_training = data_training.drop("form_id", axis=1)
data_test = data_test.drop("form_id", axis=1)

print('Training data set shape after drop: ', data_training.shape)
print('Training data set shape after drop: ', data_test.shape)


# Exploring numerical attributes
# Some numeric attributes having outliers (dti, annual_income)
# For some models it is crucial to treat ouliers.
# Classification using tree based methods is not really sensitive to outliers
# As it now we will not treat otliers (Maybe after fitting model)

numerical_attributes = data_training.select_dtypes(include=[np.number]).columns
#for num_attr in numerical_attributes:
#    # training data 
#    data_training[num_attr].hist(bins=100)
#    plt.title('training '+num_attr)
#    plt.show()
#    
#    # testing data
#    data_test[num_attr].hist(bins=100)
#    plt.title('test '+num_attr)
#    plt.show()


# Exploring categorical attributes
#
# esource need to be considered as categorical attribute not as continues variable
# esource is just a name for channel (I guess)
# this is tricky for a candidate outside of the LT 
# we should think maybe to have an explanation of this field in pdf file 
# or we can keep it as it is and see what will candidate do
# 

print("Training data set:")
print()
categorical_attributes = data_training.select_dtypes(exclude=[np.number]).columns
for cat_attr in categorical_attributes:
    print("Column: ", cat_attr)
    print(data_training[cat_attr].unique())

print()
print("Test data set:")
print()
categorical_attributes = data_test.select_dtypes(exclude=[np.number]).columns
for cat_attr in categorical_attributes:
    print("Column: ", cat_attr)
    print(data_test[cat_attr].unique())


# Some categorical values are mixied upper and lower case
# it is important to make everything same: upper or lower case

# Prepare to clean data
clean_data_training = data_training.copy()
clean_data_test = data_test.copy()

# Applying upper case to all categorical columns
for cat_attr in categorical_attributes:
    clean_data_training[cat_attr] = clean_data_training[cat_attr].str.upper()
    clean_data_test[cat_attr] = clean_data_test[cat_attr].str.upper()

# Only loan-state has nan values for test and training data sets. 
# fillna with 'UNK' value
clean_data_training['loan-state'] = clean_data_training['loan-state'].fillna(value='UNK')
clean_data_test['loan-state'] = clean_data_test['loan-state'].fillna(value='UNK')
    
# credit_score column has nan values: fillna wih 0
clean_data_training['credit_score'] = clean_data_training['credit_score'].fillna(value=0)
clean_data_test['credit_score'] = clean_data_test['credit_score'].fillna(value=0)

# Transforming esource to str type (Maybe this is not really important need to think more about it)
clean_data_training['esource'] = clean_data_training['esource'].astype(str)
clean_data_test['esource'] = clean_data_test['esource'].astype(str)

explore_raw_data(clean_data_training)


explore_raw_data(clean_data_test)


# separate target column from data
target = pd.DataFrame(clean_data_training['personal_loan_purpose_code'])
target.head()


x = 'personal_loan_purpose_code'
target_le = LabelEncoder()
target_le.fit(target[x])
target[x] = target_le.transform(target[x])

# drop 'personal_loan_purpose_code' from training data
clean_data_training.drop('personal_loan_purpose_code', 1, inplace=True)

target.head()


# We will use LabelEncoder to encode labels 
# To train LabelEncoder we will use both trainng and test data sets to avoid 
# issue of unseen categorical values: value present in test data set and not present in training data set
# Ideally we need to add raw of "UNSEEN" to each categorical attributes in training set 

original_cat_attrs = clean_data_training.select_dtypes(exclude=[np.number]).columns


# column tuples:  categorical
column_tuples = [(col, [LabelEncoder()]) for col in original_cat_attrs]

# DataFrameMapper
categorical_encoder = DataFrameMapper(column_tuples, input_df=True, df_out=False)

# fit
categorical_data = pd.concat([clean_data_training.select_dtypes(exclude=[np.number]), 
                              clean_data_test.select_dtypes(exclude=[np.number])])
#categorical_data = clean_data_training.select_dtypes(exclude=[np.number])


explore_raw_data(categorical_data)


fitted_categorical_encoder = categorical_encoder.fit(categorical_data)

# save fitted DataFrameMapper
#with open('../pkl/CategoricalMapper.pkl', 'wb') as pkl:
#    pickle.dump(fitted_categorical_mapper, pkl)


explore_raw_data(clean_data_training)


# process data for training using our encoder

train_data = pd.DataFrame(fitted_categorical_encoder.transform(clean_data_training.select_dtypes(exclude=[np.number])), 
                          columns=original_cat_attrs)
#train_data = train_data.reindex(clean_data_training.index.values)
#clean_data_training.index.values
explore_raw_data(train_data)


# Bring the cleaned up numerical data and categorical data together so that we can create a train test split

imputed_DF = clean_data_training.select_dtypes(include=[np.number])

print('imputed_DF.shape', imputed_DF.shape)


imputed_DF.columns = clean_data_training.select_dtypes(include=[np.number]).columns
#train_data = train_data.reindex(imputed_DF.index.values)
train_data = train_data.set_index(imputed_DF.index)
train_data.head()


explore_raw_data(train_data)



imputed_DF = imputed_DF.merge(pd.DataFrame(train_data), 
                              how='left', 
                              left_index=True, 
                              right_index=True
                             )


explore_raw_data(train_data)

imputed_DF.head()

explore_raw_data(imputed_DF)


X_train, X_test, y_train, y_test = model_selection.train_test_split(imputed_DF, target, test_size=0.2, random_state=0)

xgb_model = xgb.XGBClassifier(objective="multi:softprob", random_state=42,eval_metric="mlogloss",tree_method='hist',
                              eval_set=[(X_test, y_test)],n_estimators=1000,early_stopping_rounds=100,
                              n_jobs=-1, print_evaluation=100)


xgbres = xgb.XGBClassifier(objective='multi:softprob',
                           n_estimators=2000, 
                           learning_rate=0.01, 
                           gamma=0, 
                           subsample=0.50,
                           colsample_bytree=1, 
                           max_depth=7, 
                           n_jobs=-1, 
                           verbosity=2,
                           verbose_eval=10)

eval_set = [(X_test, y_test)]

xgbres.fit(X_train, y_train, eval_metric=['mlogloss'], eval_set=eval_set, early_stopping_rounds=100, verbose=10)
print(xgbres)

#xgb.plot_importance(xgbres, max_num_features=20)
#evals_result = xgbres.evals_result()
#print('Plotting metrics recorded during training...')
#ax = lightgbm.plot_metric(evals_result, metric='mlogloss')
#plt.show()
#print(xgbres)



# make predictions for Train
expected_y  = y_train
predicted_y = xgbres.predict(X_train)
    
# summarize the fit of the model
print(); print(xgbres)

print(); print('XGBoost: ')
print(); print(metrics.classification_report(expected_y, 
                                             predicted_y, 
                                             target_names=target_le.classes_))
print(); print(metrics.confusion_matrix(expected_y, 
                                        predicted_y))    

# make predictions for Test
expected_y  = y_test
predicted_y = xgbres.predict(X_test)
    
# summarize the fit of the model
print(); print(xgbres)

print(); print('XGBoost: ')
print(); print(metrics.classification_report(expected_y, 
                                             predicted_y, 
                                             target_names=target_le.classes_))
print(); print(metrics.confusion_matrix(expected_y, 
                                        predicted_y))    


# process data for scoring using our encoder
data_test_cat = pd.DataFrame(fitted_categorical_encoder.transform(clean_data_test.select_dtypes(exclude=[np.number])), 
                          columns=original_cat_attrs)
data_test_cat.head()


# Bring the cleaned up numerical data and categorical data together so that we can score

data_for_scoring = clean_data_test.select_dtypes(include=[np.number])

print('data_for_scoring.shape', data_for_scoring.shape)

data_for_scoring.columns = clean_data_training.select_dtypes(include=[np.number]).columns


data_test_cat = data_test_cat.set_index(data_for_scoring.index)
print(train_data.head())


data_for_scoring = data_for_scoring.merge(pd.DataFrame(data_test_cat), 
                                          how='left', 
                                          left_index=True, 
                                          right_index=True)
data_for_scoring.head()


# scoring test data set

ScoredArray = xgbres.predict_proba(data_for_scoring)


columnNames = target_le.classes_



personal_loan_array = xgbres.predict(data_for_scoring)
best_personal_loan = target_le.inverse_transform(personal_loan_array.astype(float).astype(int))
best_personal_loan



ScoredDf = pd.DataFrame(best_personal_loan,columns=['predictedLabel'])
ScoredDf = ScoredDf.merge(pd.DataFrame(ScoredArray, columns=columnNames), how='left', left_index=True, right_index=True)
ScoredDf = ScoredDf.set_index(data_for_scoring.index)


ScoredDf = ScoredDf.round(3)
ScoredDf.head()


# uploading scored data farme to csv
ScoredDf.to_csv('predictions.csv')





