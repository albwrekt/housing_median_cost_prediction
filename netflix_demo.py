#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 23 21:26:42 2020

@author: albwrekt
"""

import os
import pandas as pd
import numpy as np
from pandas.plotting import scatter_matrix


# this dataset will be used to predict the rating of the movie

DATA_PATH = "../../archive/netflix_titles.csv"
INDEX = "show_id"

# method for reading in the data
def load_data(datapath=DATA_PATH):
    return pd.read_csv(datapath)

# split up the training set and the testing set
def split_test_train_set(testset, test_ratio):
    shuffled_indices = np.random.permutation(len(testset))
    test_set_size = int(len(testset) * test_ratio)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    return testset.iloc[train_indices], testset.iloc[test_indices]

def split_numbers_and_categories(dataset, index=INDEX):
    num_list = dataset.copy()
    cat_list = dataset.copy()
    for key in dataset.keys():
        if dataset[key].dtype == 'object':
            num_list.drop(key, axis=1, inplace=True)
        else:
            cat_list.drop(key, axis=1, inplace=True)
    max_data_count = len(dataset[index])
    return max_data_count, num_list, cat_list

def process_numeric_dataset(num_set):
    num_corr = num_set.corr()
    for key in num_set.keys():
        print(num_corr[key].sort_values(ascending=False))
        print("\n\n")
    scatter_matrix(num_set, figsize=(12, 8))
    
def process_category_datasets(cat_set):
    print("category somethign here")

# investigate the dataset and display the information to the user
def investigate_dataset(data):
    print(data.head())
    print(data.info())
    print(data.describe())
    print(data.keys())
    for key in data.keys():
        value_counts = data[key].value_counts().to_frame()
        print("Overall Key:", key)
        for value in value_counts:
            print(type(value))
            print(value)
    
    
dataset = load_data()
train_set, test_set = split_test_train_set(dataset, 0.3)
max_data_count, num_list, cat_list = split_numbers_and_categories(train_set)
process_numeric_dataset(num_list)




    