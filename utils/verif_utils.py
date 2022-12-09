# -*- coding: utf-8 -*-

import sys
sys.path.append('../')
# numpy and pandas for data manipulation
from time import time
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')
import os.path
from random import randrange
#import tensorflow as tf
from sklearn.pipeline import make_pipeline, Pipeline, FeatureUnion
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder, KBinsDiscretizer, Normalizer, MaxAbsScaler, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV, KFold, cross_val_score, train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectKBest, SelectPercentile, VarianceThreshold

#from utils.ml_fairness import *
from utils.standard_data import *

from tensorflow.keras.models import load_model
from tensorflow.keras import backend as K
from z3 import *
import numpy as np 
import pandas as pd 
from sklearn.metrics import accuracy_score,confusion_matrix,roc_auc_score
from sklearn import metrics
import collections
import time
import datetime

def load_adult_adf():
    #!/usr/bin/env python
    # coding: utf-8
    
    # data = pd.read_csv("adult.csv")
    train_path = '../../data/adult/adult.data'
    test_path = '../../data/adult/adult.test'
    
    column_names = ['age', 'workclass', 'fnlwgt', 'education',
                'education-num', 'marital-status', 'occupation', 'relationship',
                'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week',
                'native-country', 'income-per-year']
    na_values=['?']
    
    train = pd.read_csv(train_path, header=None, names=column_names, 
                        skipinitialspace=True, na_values=na_values)
    test = pd.read_csv(test_path, header=0, names=column_names,
                       skipinitialspace=True, na_values=na_values)
    
    df = pd.concat([test, train], ignore_index=True)
    
    del_cols = ['fnlwgt'] # 'education-num'
    df.drop(labels = del_cols,axis = 1,inplace = True)
    
    ##### Drop na values
    dropped = df.dropna()
    count = df.shape[0] - dropped.shape[0]
    print("Missing Data: {} rows removed.".format(count))
    df = dropped
    
    cat_feat = ['sex', 'workclass', 'education', 'marital-status', 'occupation', 'relationship', 'native-country']
    ## Implement label encoder instead of one-hot encoder
    for feature in cat_feat:
        le = LabelEncoder()
        df[feature] = le.fit_transform(df[feature])

#    df = pd.get_dummies(df, columns=cat_feat, prefix_sep='=')
    
    ## Implement label encoder instead of one-hot encoder
    cat_feat = ['race']
    for feature in cat_feat:
        le = LabelEncoder()
        df[feature] = le.fit_transform(df[feature])
        
    
    bin_cols = ['capital-gain', 'capital-loss']
    for feature in bin_cols:
        bins = KBinsDiscretizer(n_bins=20, encode='ordinal', strategy='uniform')
        df[feature] = bins.fit_transform(df[[feature]])

    
#    df = df[columns]
    label_name = 'income-per-year'
    
    favorable_label = 1
    unfavorable_label = 0
    favorable_classes=['>50K', '>50K.']
    
    pos = np.logical_or.reduce(np.equal.outer(favorable_classes, df[label_name].to_numpy()))
    df.loc[pos, label_name] = favorable_label
    df.loc[~pos, label_name] = unfavorable_label
    
    X = df.drop(labels = [label_name], axis = 1, inplace = False)
    y = df[label_name]
    
    y = pd.get_dummies(y, columns=[label_name], prefix_sep='=')
    
    seed = 42 # randrange(100)
#    train, test  = train_test_split(df, test_size = 0.15, random_state = seed)
    X_train, X_test, y_train, y_test  = train_test_split(X, y, test_size = 0.15, random_state = seed)        
    return (df, X_train.to_numpy(), y_train.to_numpy().astype('int'), X_test.to_numpy(), y_test.to_numpy().astype('int'))


def load_adult_ac1():
    #!/usr/bin/env python
    # coding: utf-8
    
    # data = pd.read_csv("adult.csv")
    train_path = '../../data/adult/adult.data'
    test_path = '../../data/adult/adult.test'
    
    column_names = ['age', 'workclass', 'fnlwgt', 'education',
                'education-num', 'marital-status', 'occupation', 'relationship',
                'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week',
                'native-country', 'income-per-year']
    na_values=['?']
    
    train = pd.read_csv(train_path, header=None, names=column_names, 
                        skipinitialspace=True, na_values=na_values)
    test = pd.read_csv(test_path, header=0, names=column_names,
                       skipinitialspace=True, na_values=na_values)
    
    df = pd.concat([test, train], ignore_index=True)
    
    del_cols = ['fnlwgt'] # 'education-num'
    df.drop(labels = del_cols,axis = 1,inplace = True)
    
    ##### Drop na values
    dropped = df.dropna()
    count = df.shape[0] - dropped.shape[0]
    print("Missing Data: {} rows removed.".format(count))
    df = dropped
    
    cat_feat = ['sex', 'workclass', 'education', 'marital-status', 'occupation', 'relationship', 'native-country']
    ## Implement label encoder instead of one-hot encoder
    for feature in cat_feat:
        le = LabelEncoder()
        df[feature] = le.fit_transform(df[feature])

#    df = pd.get_dummies(df, columns=cat_feat, prefix_sep='=')
    
    ## Implement label encoder instead of one-hot encoder
    cat_feat = ['race']
    for feature in cat_feat:
        le = LabelEncoder()
        df[feature] = le.fit_transform(df[feature])
        
    
    bin_cols = ['capital-gain', 'capital-loss']
    for feature in bin_cols:
        bins = KBinsDiscretizer(n_bins=20, encode='ordinal', strategy='uniform')
        df[feature] = bins.fit_transform(df[[feature]])

    
#    df = df[columns]
    label_name = 'income-per-year'
    
    favorable_label = 1
    unfavorable_label = 0
    favorable_classes=['>50K', '>50K.']
    
    pos = np.logical_or.reduce(np.equal.outer(favorable_classes, df[label_name].to_numpy()))
    df.loc[pos, label_name] = favorable_label
    df.loc[~pos, label_name] = unfavorable_label
    
    X = df.drop(labels = [label_name], axis = 1, inplace = False)
    y = df[label_name]
    
    
    seed = 42 # randrange(100)
#    train, test  = train_test_split(df, test_size = 0.15, random_state = seed)
    X_train, X_test, y_train, y_test  = train_test_split(X, y, test_size = 0.15, random_state = seed)        
    return (df, X_train.to_numpy(), y_train.to_numpy().astype('int'), X_test.to_numpy(), y_test.to_numpy().astype('int'))

def load_german():
    filepath = '../../data/german/german.data'
    column_names = ['status', 'month', 'credit_history',
                'purpose', 'credit_amount', 'savings', 'employment',
                'investment_as_income_percentage', 'personal_status',
                'other_debtors', 'residence_since', 'property', 'age',
                'installment_plans', 'housing', 'number_of_credits',
                'skill_level', 'people_liable_for', 'telephone',
                'foreign_worker', 'credit']
    na_values=[]
    df = pd.read_csv(filepath, sep=' ', header=None, names=column_names,na_values=na_values)
    df['age'] = df['age'].apply(lambda x: np.float(x >= 26))
    df = german_custom_preprocessing(df)
    feat_to_drop = ['personal_status']
    df = df.drop(feat_to_drop, axis=1)
    
    cat_feat = ['status', 'credit_history', 'purpose', 'savings', 'employment', 'other_debtors', 'property', 'installment_plans', 
                'housing', 'skill_level', 'telephone', 'foreign_worker']
    
    
    for f in cat_feat:
        label = LabelEncoder()
        df[f] = label.fit_transform(df[f])      
    
#    bin_cols = ['capital-gain', 'capital-loss']
#    for feature in bin_cols:
#        bins = KBinsDiscretizer(n_bins=20, encode='ordinal', strategy='uniform')
#        df[feature] = bins.fit_transform(df[[feature]])
    
#    df = df[columns]
    label_name = 'credit'
#    
    favorable_label = 1
    unfavorable_label = 0
    #favorable_classes=['>50K', '>50K.']
    
    
    #pos = np.logical_or.reduce(np.equal.outer(favorable_classes, df[label_name].to_numpy()))
    #df.loc[pos, label_name] = favorable_label
    #df.loc[~pos, label_name] = unfavorable_label
#    
    X = df.drop(labels = [label_name], axis = 1, inplace = False)
    y = df[label_name]
#    
#    
    seed = 42 # randrange(100)
#    train, test  = train_test_split(df, test_size = 0.15, random_state = seed)
    X_train, X_test, y_train, y_test  = train_test_split(X, y, test_size = 0.15, random_state = seed)        
    return (df, X_train.to_numpy(), y_train.to_numpy().astype('int'), X_test.to_numpy(), y_test.to_numpy().astype('int'))

def load_bank():
    file_path = '../../data/bank/bank-additional-full.csv'

    column_names = ['age', 'job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 
                    'month', 'day_of_week', 'duration', 'emp.var.rate',  
                    'campaign', 'pdays', 'previous', 'poutcome', 'y']
    na_values=['unknown']
    
    df = pd.read_csv(file_path, sep=';', na_values=na_values)
    
    ### Drop na values
    dropped = df.dropna()
    count = df.shape[0] - dropped.shape[0]
    print("Missing Data: {} rows removed.".format(count))
    df = dropped
    columns = ['education=Assoc-acdm', 'education=Assoc-voc', 'education=Bachelors',]
    
    df['age'] = df['age'].apply(lambda x: np.float(x >= 25))
    
    ## Feature selection
    # features_to_keep = []
    # df = df[features_to_keep]
    
    # Create a one-hot encoding of the categorical variables.
    cat_feat = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'day_of_week', 'poutcome']
    #df = pd.get_dummies(df, columns=cat_feat, prefix_sep='=')
    
    
    for f in cat_feat:
        label = LabelEncoder()
        df[f] = label.fit_transform(df[f])      
    
#    bin_cols = ['capital-gain', 'capital-loss']
#    for feature in bin_cols:
#        bins = KBinsDiscretizer(n_bins=20, encode='ordinal', strategy='uniform')
#        df[feature] = bins.fit_transform(df[[feature]])
    #print(df.columns)
    
    df = df[column_names]
    label_name='y'
    favorable_label = 1
    unfavorable_label = 0
    favorable_classes=['yes']
    
    pos = np.logical_or.reduce(np.equal.outer(favorable_classes, df[label_name].to_numpy()))
    df.loc[pos, label_name] = favorable_label
    df.loc[~pos, label_name] = unfavorable_label
    df = df.round(0).astype(int)
    
#    
    X = df.drop(labels = [label_name], axis = 1, inplace = False)
    y = df[label_name]
#    
#    
    seed = 42 # randrange(100)
#    train, test  = train_test_split(df, test_size = 0.15, random_state = seed)
    X_train, X_test, y_train, y_test  = train_test_split(X, y, test_size = 0.15, random_state = seed)        
    return (df, X_train.to_numpy(), y_train.to_numpy().astype('int'), X_test.to_numpy(), y_test.to_numpy().astype('int'))


def load_adult():
    #!/usr/bin/env python
    # coding: utf-8
    
    # data = pd.read_csv("adult.csv")
    train_path = '../../data/adult/adult.data'
    test_path = '../../data/adult/adult.test'
    
    column_names = ['age', 'workclass', 'fnlwgt', 'education',
                'education-num', 'marital-status', 'occupation', 'relationship',
                'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week',
                'native-country', 'income-per-year']
    na_values=['?']
    
    train = pd.read_csv(train_path, header=None, names=column_names, 
                        skipinitialspace=True, na_values=na_values)
    test = pd.read_csv(test_path, header=0, names=column_names,
                       skipinitialspace=True, na_values=na_values)
    
    df = pd.concat([test, train], ignore_index=True)
    
    imputer = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
    df['workclass'] = imputer.fit_transform(df[['workclass']]).ravel()
    df['occupation'] = imputer.fit_transform(df[['occupation']]).ravel()
    df['native-country'] = imputer.fit_transform(df[['native-country']]).ravel()
    
    hs_grad = ['HS-grad','11th','10th','9th','12th']
    elementary = ['1st-4th','5th-6th','7th-8th']
    
    df['education'].replace(to_replace = hs_grad,value = 'HS-grad',inplace = True)
    df['education'].replace(to_replace = elementary,value = 'elementary_school',inplace = True)
    
    married= ['Married-spouse-absent','Married-civ-spouse','Married-AF-spouse']
    separated = ['Separated','Divorced']
    
    #replace elements in list.
    df['marital-status'].replace(to_replace = married ,value = 'Married',inplace = True)
    df['marital-status'].replace(to_replace = separated,value = 'Separated',inplace = True)
    
    self_employed = ['Self-emp-not-inc','Self-emp-inc']
    govt_employees = ['Local-gov','State-gov','Federal-gov']
    
    #replace elements in list.
    df['workclass'].replace(to_replace = self_employed ,value = 'Self_employed',inplace = True)
    df['workclass'].replace(to_replace = govt_employees,value = 'Govt_employees',inplace = True)
    
    del_cols = ['education-num', 'fnlwgt']
    df.drop(labels = del_cols,axis = 1,inplace = True)
    
    ##### Drop na values
    dropped = df.dropna()
    count = df.shape[0] - dropped.shape[0]
#    print("Missing Data: {} rows removed.".format(count))
    df = dropped
    
    cat_feat = ['sex', 'workclass', 'education', 'marital-status', 'occupation', 'relationship', 'native-country'] # 'race'
    ## One-hot encoding of categorical features
    df = pd.get_dummies(df, columns=cat_feat, prefix_sep='=')
    
    ## Implement label encoder instead of one-hot encoder
    cat_feat = ['race']
    for feature in cat_feat:
        le = LabelEncoder()
        df[feature] = le.fit_transform(df[feature])
    
    
    columns = ['education=Assoc-acdm', 'education=Assoc-voc', 'education=Bachelors',
           'education=Doctorate', 'education=HS-grad', 'education=Masters',
           'education=Preschool', 'education=Prof-school',
           'education=elementary_school', 'sex=Female',
           'marital-status=Married', 'marital-status=Separated',
           'marital-status=Widowed', 'occupation=Adm-clerical',
           'occupation=Armed-Forces', 'occupation=Craft-repair',
           'occupation=Exec-managerial', 'occupation=Farming-fishing',
           'occupation=Handlers-cleaners', 'occupation=Machine-op-inspct',
           'occupation=Priv-house-serv', 'occupation=Prof-specialty',
           'occupation=Protective-serv', 'occupation=Sales',
           'occupation=Tech-support', 'occupation=Transport-moving', 
           'relationship=Husband', 'relationship=Not-in-family',
           'relationship=Other-relative', 'relationship=Own-child',
           'relationship=Unmarried', 'relationship=Wife',
           'workclass=Govt_employees', 'workclass=Never-worked',
           'workclass=Private', 'workclass=Self_employed', 'workclass=Without-pay',
           'race', #'race=Amer-Indian-Eskimo', 'race=Asian-Pac-Islander', 'race=Black', 'race=Other', 'race=White',
           'age', 'capital-gain', 'capital-loss', 'hours-per-week', 'income-per-year']
    
    df = df[columns]
    label_name = 'income-per-year'
    
    favorable_label = 1
    unfavorable_label = 0
    favorable_classes=['>50K', '>50K.']
    
    pos = np.logical_or.reduce(np.equal.outer(favorable_classes, df[label_name].to_numpy()))
    df.loc[pos, label_name] = favorable_label
    df.loc[~pos, label_name] = unfavorable_label
    
    X = df.drop(labels = [label_name], axis = 1, inplace = False)
    y = df[label_name]
    
    seed = 42 # randrange(100)
#    train, test  = train_test_split(df, test_size = 0.15, random_state = seed)
    X_train, X_test, y_train, y_test  = train_test_split(X, y, test_size = 0.15, random_state = seed)
    
#    pro_att_name = ['race'] # ['race', 'sex']
#    priv_class = ['White'] # ['White', 'Male']
#    reamining_cat_feat = []
#    data_train, X_train, y_train = load_adult_data(train, pro_att_name, priv_class, reamining_cat_feat)
#    data_test, X_test, y_test = load_adult_data(test, pro_att_name, priv_class, reamining_cat_feat)
#    train_dataframe = data_train.convert_to_dataframe()[0]
#    test_dataframe = data_test.convert_to_dataframe()[0]    
#    return (df, X_train, y_train, X_test, y_test)
        
    return (df, X_train.to_numpy(), y_train.to_numpy().astype('int'), X_test.to_numpy(), y_test.to_numpy().astype('int'))

#df, X_train, y_train, X_test, y_test = load_adult()

def load__trained_model(path):
   model = load_model(path)
   return model

def get_layer_weights(model):
    names = []
    weights = []
    biases = []
    
    for layer in model.layers:
        names.append(layer.get_config().get('name'))
        weights.append(layer.get_weights()[0])
        biases.append(layer.get_weights()[1])
    return (names, weights, biases)

def get_layer_outputs(model, single_input):
    inp = model.input                                           # input placeholder
    outputs = [layer.output for layer in model.layers]          # all layer outputs
    functors = [K.function([inp], [out]) for out in outputs]    # evaluation functions
    X = single_input.reshape(1, 42)
    layer_outs = [func([X]) for func in functors]
    return layer_outs

def single_predict(single_input):
    model.predict(X)
    return (model.predict(X) > 0.5)

def print_cols(dataframe):
    for col in dataframe:
        index = dataframe.columns.get_loc(col)
        print(index, col)

def print_uniques(dataframe):
    for col in dataframe:
        print(dataframe[col].unique())

def relu(x):
    return np.maximum(0, x)

def z3Relu(x):
    #return np.vectorize(lambda y: If(y >= 0 , y, RealVal(0)))(x)
    #FPVal(2.5, FPSort(8, 24))
    return np.vectorize(lambda y: If(y >= 0 , y, RealVal(0)))(x)

def z3Abs(x):
    return If(x <= 0, -x, x)

def bigAnd(l):
    if l == []:
        return True
    if len(l) == 1:
        return l[0]
    return And(*l)

def bigOr(l):
    if l == []:
        return False
    if len(l) == 1:
        return l[0]
    return Or(*l)

def z3ToFrac(r):
    assert(is_rational_value(r))
    return r.as_fraction()

def z3ToFloat(r):
    #this is sometimes causing an error
    #"OverflowError: long int too large to convert to float"
    return float(r.as_decimal(100).strip('?'))

def z3ToMath(f):
    s = str(f)
    s = s.replace("(", "[")
    s = s.replace(")", "]")
    return s

def compute_time(start_time, current_time):
    runtime = current_time - start_time
    # print(round(runtime, 2))
    return round(runtime, 2)

def print_metadata(dataframe):
    max = dataframe.max()
    min = dataframe.min()
    print('@@@@@ Max: ')
    print(max)
    print('@@@@@ Min: ')
    print(min)
    print('@@@@@ Uniques: ')
    for col in dataframe:
        print(col)
        print(dataframe[col].unique())

# In[]
#print_cols(train_dataframe)
        
def in_const_domain_ac1(df, x, x_, ranges, default):
    dataframe = df.drop(df.columns[len(df.columns)-1], axis=1, inplace=False)
    props = []
    
    for col in dataframe:
        var = ''
        for var_name in ranges.keys():
            if(col.startswith(var_name)):
                var = col
                break
            
        index = dataframe.columns.get_loc(col)
        if(var != ''):
            props.append(And(x[index] >= ranges[var_name][0], x[index] <= ranges[var_name][1]))
            props.append(And(x_[index] >= ranges[var_name][0], x_[index] <= ranges[var_name][1]))
        else:
            props.append(And(x[index] >= default[0], x[index] <= default[1]))
            props.append(And(x_[index] >= default[0], x_[index] <= default[1]))
    return props

def in_const_range(df, x, x_, var_name, lb, ub):
    dataframe = df.drop(df.columns[len(df.columns)-1], axis=1, inplace=False)
    props = []
    for col in dataframe:
        if col == var_name:
            index = dataframe.columns.get_loc(col)
            props.append(And(x[index] <= lb, x[index] >= ub))
            props.append(And(x_[index] <= lb, x_[index] >= ub))
    return props

def in_const_equality_domain(df, x, x_, ranges, PA):
    dataframe = df.drop(df.columns[len(df.columns)-1], axis=1, inplace=False)
    props = []
    
    for col in dataframe:
        index = dataframe.columns.get_loc(col)
            # props.append(And(x[index] >= ranges[var_name][0], x[index] <= ranges[var_name][1]))
            # props.append(And(x_[index] >= ranges[var_name][0], x_[index] <= ranges[var_name][1]))

        if(col in PA):
            p1 = []
            p2 = []
            val = ranges[col][0]
            while 1:
                if val > ranges[col][1]:
                    break
                p1.append(x[index] == val)
                p2.append(x_[index] == val)
                val += 1
            props.append(bigOr(p1))
            props.append(bigOr(p2))

        else:    
            p = []
            val = ranges[col][0]
            while 1:
                if val > ranges[col][1]:
                    break
                
                p.append(x[index] == val)
                #p.append(x_[index] == val)
                val += 1
#            print(col)
#            print(bigOr(p))
            props.append(bigOr(p))

    return props

def in_const_domain_bank(df, x, x_, ranges, PA):
    label_name = 'y'
    dataframe = df.drop(labels = [label_name], axis=1, inplace=False)
    #dataframe = df.drop(df.columns[len(df.columns)-1], axis=1, inplace=False)
    props = []
    
    for col in dataframe:
        index = dataframe.columns.get_loc(col)
            
        if(col in PA):
            props.append(And(x[index] >= ranges[col][0], x[index] <= ranges[col][1]))
            props.append(And(x_[index] >= ranges[col][0], x_[index] <= ranges[col][1]))            

        else:
             props.append(And(x[index] >= ranges[col][0], x[index] <= ranges[col][1]))
             #props.append(And(x_[index] >= ranges[col][0], x_[index] <= ranges[col][1])) # this ones could be ommited

    return props

def in_const_bank(df, x, var_name, op, rhs):
    label_name = 'y'
    dataframe = df.drop(labels = [label_name], axis=1, inplace=False)
    props = []
    for col in dataframe:
        if col == var_name:
            index = dataframe.columns.get_loc(col)
            if(isinstance(rhs, int) or isinstance(rhs, float)):
                right = rhs
            else:
                right = rhs[index]
                
            if(op == 'gt'):
                props.append(x[index] > right)
            elif(op == 'lt'):
                props.append(x[index] < right)
            elif(op == 'gte'):
                props.append(x[index] >= right)
            elif(op == 'lte'):
                props.append(x[index] <= right)
            elif(op == 'eq'):
                props.append(x[index] == right)
            elif(op == 'neq'):
                props.append(x[index] != right)
            else:
                raise Exception('The operand is not defined!') 
    return props

def in_const_german(df, x, var_name, op, rhs):
    label_name = 'credit'
    dataframe = df.drop(labels = [label_name], axis=1, inplace=False)
    props = []
    for col in dataframe:
        if col == var_name:
            index = dataframe.columns.get_loc(col)
            if(isinstance(rhs, int) or isinstance(rhs, float)):
                right = rhs
            else:
                right = rhs[index]
                
            if(op == 'gt'):
                props.append(x[index] > right)
            elif(op == 'lt'):
                props.append(x[index] < right)
            elif(op == 'gte'):
                props.append(x[index] >= right)
            elif(op == 'lte'):
                props.append(x[index] <= right)
            elif(op == 'eq'):
                props.append(x[index] == right)
            elif(op == 'neq'):
                props.append(x[index] != right)
            else:
                raise Exception('The operand is not defined!') 
    return props

def in_const_domain_german(df, x, x_, ranges, PA):
    label_name = 'credit'
    dataframe = df.drop(labels = [label_name], axis=1, inplace=False)
    #dataframe = df.drop(df.columns[len(df.columns)-1], axis=1, inplace=False)
    props = []
    
    for col in dataframe:
        index = dataframe.columns.get_loc(col)
            
        if(col in PA):
            props.append(And(x[index] >= ranges[col][0], x[index] <= ranges[col][1]))
            props.append(And(x_[index] >= ranges[col][0], x_[index] <= ranges[col][1]))            

        else:
             props.append(And(x[index] >= ranges[col][0], x[index] <= ranges[col][1]))
             #props.append(And(x_[index] >= ranges[col][0], x_[index] <= ranges[col][1])) # this ones could be ommited

    return props

def in_const_adult(df, x, var_name, op, rhs):
    label_name = 'income-per-year'
    dataframe = df.drop(labels = [label_name], axis=1, inplace=False)
    props = []
    for col in dataframe:
        if col == var_name:
            index = dataframe.columns.get_loc(col)
            if(isinstance(rhs, int) or isinstance(rhs, float)):
                right = rhs
            else:
                right = rhs[index]
                
            if(op == 'gt'):
                props.append(x[index] > right)
            elif(op == 'lt'):
                props.append(x[index] < right)
            elif(op == 'gte'):
                props.append(x[index] >= right)
            elif(op == 'lte'):
                props.append(x[index] <= right)
            elif(op == 'eq'):
                props.append(x[index] == right)
            elif(op == 'neq'):
                props.append(x[index] != right)
            else:
                raise Exception('The operand is not defined!') 
    return props

def in_const_domain_adult(df, x, x_, ranges, PA):
    label_name = 'income-per-year'
    dataframe = df.drop(labels = [label_name], axis=1, inplace=False)
    #dataframe = df.drop(df.columns[len(df.columns)-1], axis=1, inplace=False)
    props = []
    
    for col in dataframe:
        index = dataframe.columns.get_loc(col)
            
        if(col in PA):
            props.append(And(x[index] >= ranges[col][0], x[index] <= ranges[col][1]))
            props.append(And(x_[index] >= ranges[col][0], x_[index] <= ranges[col][1]))            

        else:
             props.append(And(x[index] >= ranges[col][0], x[index] <= ranges[col][1]))
             #props.append(And(x_[index] >= ranges[col][0], x_[index] <= ranges[col][1])) # this ones could be ommited

    return props

def in_const_diff_adult(df, x, x_, var_name, threshold):
    label_name = 'income-per-year'
    dataframe = df.drop(labels = [label_name], axis=1, inplace=False)
    props = []
    for col in dataframe:
        if col == var_name:
            index = dataframe.columns.get_loc(col)  
            props.append(z3Abs(x[index] - x_[index]) <= threshold)
    return props

def in_const_diff_german(df, x, x_, var_name, threshold):
    label_name = 'credit'
    dataframe = df.drop(labels = [label_name], axis=1, inplace=False)
    props = []
    for col in dataframe:
        if col == var_name:
            index = dataframe.columns.get_loc(col)  
            props.append(z3Abs(x[index] - x_[index]) <= threshold)
    return props

def in_const_diff_bank(df, x, x_, var_name, threshold):
    label_name = 'y'
    dataframe = df.drop(labels = [label_name], axis=1, inplace=False)
    props = []
    for col in dataframe:
        if col == var_name:
            index = dataframe.columns.get_loc(col)  
            props.append(z3Abs(x[index] - x_[index]) <= threshold)
    return props

def in_const_diff(df, x, x_, var_name, op, threshold):
    dataframe = df.drop(df.columns[len(df.columns)-1], axis=1, inplace=False)
    props = []
    for col in dataframe:
        if col == var_name:
            index = dataframe.columns.get_loc(col)
            if(op == 'gt'):
                props.append(z3Abs(x[index] - x_[index]) > threshold)
            elif(op == 'lt'):
                props.append(z3Abs(x[index] - x_[index]) < threshold)
            elif(op == 'gte'):
                props.append(z3Abs(x[index] - x_[index]) >= threshold)
            elif(op == 'lte'):
                props.append(z3Abs(x[index] - x_[index]) <= threshold)
            elif(op == 'eq'):
                props.append(z3Abs(x[index] - x_[index]) == threshold)
            elif(op == 'neq'):
                props.append(z3Abs(x[index] - x_[index]) != threshold)
            else:
                raise Exception('The operand is not defined!') 
    return props


def in_const_equals(df, x, x_, inequality):
    dataframe = df.drop(df.columns[len(df.columns)-1], axis=1, inplace=False)
    
    ignore_indexes = []
    for ineq in inequality:
        for col in dataframe:
            if(col.startswith(ineq)):
                ignore_indexes.append(dataframe.columns.get_loc(col))
    props = []
    for col in dataframe:
        index = dataframe.columns.get_loc(col)
        if(index not in ignore_indexes):
            props.append(x[index] == x_[index])
    return props



def in_const_single(df, x, var_name, op, rhs):
    rhs = rhs.item()
    dataframe = df.drop(df.columns[len(df.columns)-1], axis=1, inplace=False)
    props = []
    for col in dataframe:
        if(col == var_name):
            index = dataframe.columns.get_loc(col)
            if(isinstance(rhs, int) or isinstance(rhs, float)):
                right = rhs
            else:
                print(rhs)
                print(type(rhs))
                raise Exception('Right hand side type is not handled!')     
            if(op == 'gt'):
                props.append(x[index] > right)
            elif(op == 'lt'):
                props.append(x[index] < right)
            elif(op == 'gte'):
                props.append(x[index] >= right)
            elif(op == 'lte'):
                props.append(x[index] <= right)
            elif(op == 'eq'):
                props.append(x[index] == right)
            elif(op == 'neq'):
                props.append(x[index] != right)
            else:
                raise Exception('The operand is not defined!')
    return props

def cols_starts_with(df, col_name):
    dataframe = df.drop(df.columns[len(df.columns)-1], axis=1, inplace=False)
    cols = []
    for col in dataframe:
        if(col.startswith(col_name)):
            cols.append(col)
    return cols

def unique_vals(df, col_name):
    dataframe = df.drop(df.columns[len(df.columns)-1], axis=1, inplace=False)
    for col in dataframe:
        if(col == col_name):
            return dataframe[col].unique()


def parse_z3Model(m):
    ce_x = {}
    ce_x_ = {}
    for d in m:

        variable = str(d)
        
        val = str(m[d])
#        val = m[d].as_decimal(8)
#        if val.endswith('?'):
#            val = val[:-1]
            
        if(str(d).startswith('x_')):
            ce_x_[int(variable[2:])] = val
        else:
            ce_x[int(variable[1:])] = val

    ce_x = dict(sorted(ce_x.items()))
    ce_x_ = dict(sorted(ce_x_.items()))
    
    inp1 = list(ce_x.values())
    inp2 = list(ce_x_.values())
    
#    for i in range(len(inp1)):
#        if inp[0].contains('/'):
#            print float(r.numerator_as_long())/float(r.denominator_as_long())
    
    return (inp1, inp2)


def sigmoid(x):
    # y = 1 / (1 + math.exp(-x3)) # WP computer for sigmoid
    y = 1 / (1 + math.exp(-x))
    return y

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

def get_y_pred(net, w, b, X_test):
    y_all = []
    for x in X_test:
        y = net(x, w, b)
        res = sigmoid(y)
        y_pred = res > 0.5
        y_all.append(y_pred)
    return y_all

def y_pred_mismatch(net, w, b, pr_w, pr_b, X_test, y_test):
    y_all = []
    for x in X_test:
        y = net(x, w, b)
        res = sigmoid(y)
        
        y_ = net(x, pr_w, pr_b)
        res_ = sigmoid(y_)
        
        y_pred = res > 0.5
        y_pred_ = res_ > 0.5
        if y_pred != y_pred_:
            print('MISMATCH')
            print(res)
            print(res_)