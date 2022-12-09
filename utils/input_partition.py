#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
sys.path.append('../../')
# numpy and pandas for data manipulation
from random import randrange
import numpy as np 
import pandas as pd
import collections
import time
import datetime
import copy
import itertools
from utils.verif_utils import *

def partition(r_dict, partition_size):
    partition_dict = {}
    
    cols = r_dict.keys()
    for col in cols:
        low = r_dict[col][0]
        high = r_dict[col][1]
        col_size = high - low + 1
        
        if col_size <= partition_size:
            continue
        
        cur_low = low
        cur_high = low + partition_size - 1
        parts = []
        
        while 1:
            part = [cur_low, cur_high]
            if(cur_low > high):
                break
            if(cur_high >= high):
                part = [cur_low, high]
                parts.append(part)
                partition_dict[col] = parts
                break
            parts.append(part)
            cur_low = cur_high + 1
            cur_high = cur_high + partition_size
    
    return partition_dict

def partitioned_ranges(A, PA, p_dict, range_dict):
    new_ranges = {}
    for attr in A:
        #if attr not in PA:
        if attr not in p_dict.keys():
            new_ranges[attr] = range_dict[attr]
    
    parts = [] # for each partition attrs, one element. each element has mupltiple partitions
    for p_attr in p_dict.keys():
        parts.append(p_dict[p_attr])
    
    combs = list(itertools.product(*parts)) # all combinations
    
    # distribute combinations
    total = 0
    partition_list = []
    for comb in combs:
        partitioned = copy.deepcopy(new_ranges) #new_ranges.copy()
        index = 0
        
        for p_attr in p_dict.keys():
    
            partitioned[p_attr] = comb[index]
            index += 1
        
        #print(partitioned) # # One partition of new_ranges completed
        partition_list.append(partitioned)
        total += 1
    return partition_list

def p_list_density(range_dict, p_list, df):
    label_name = 'income-per-year'
    data = df.drop(labels = [label_name], axis=1, inplace=False)
    total_count = df.shape[0]
    
    p_list_counts = [0] * len(p_list)
    p_density = []
    
    ordered_cols = range_dict
    for col in range_dict.keys():
        ordered_cols[col] = range_dict[col][1] - range_dict[col][0] + 1
    ordered_cols = {k: v for k, v in sorted(ordered_cols.items(), key=lambda item: item[1])}
    
    
    for index, row in data.iterrows():
        i = 0
        for p in p_list:
            
                
            outside = False
            for col in ordered_cols.keys(): 
                      
                if row[col] < p[col][0] or row[col] > p[col][1]:
                    outside = True
                    break
            
            if not outside:
                p_list_counts[i] += 1
            i += 1
    
    for c in p_list_counts:
        prob = c/total_count
        p_density.append(prob)
            
    return p_density

    

## Z3 Essentials


#print(s.assertions())
#print(s.units())
#print(s.non_units())
#print(s.sexpr())
#print(s.proof())

#print(m)
#print(s.assertions) 

#s.set("produce-proofs", True)