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
import copy

def z3Relu(x):
    return np.vectorize(lambda y: If(y >= 0 , y, 0))(x)

def z3_layer1_ws_net(x, w, b):
    fl_x = np.array([Real('fl_x%s' % i) for i in range(13)])
    for i in range(len(x)):
        fl_x[i] = ToReal(x[i])     
    x1 = w[0].T @ fl_x + b[0]
    return x1
def z3_layer2_ws_net(x, w, b): 
    x2 = w[1].T @ x + b[1]
    return x2
def z3_layer3_ws_net(x, w, b): 
    x3 = w[2].T @ x + b[2]
    return x3
def z3_layer4_ws_net(x, w, b):
    x4 = w[3].T @ x + b[3]
    return x4
def z3_layer5_ws_net(x, w, b):     
    x5 = w[4].T @ x + b[4]
    return x5
def z3_layer6_ws_net(x, w, b):     
    x6 = w[5].T @ x + b[5]
    return x6
def z3_layer7_ws_net(x, w, b):     
    x7 = w[6].T @ x + b[6]
    return x7
def z3_layer8_ws_net(x, w, b):     
    x8 = w[7].T @ x + b[7]
    return x8
def z3_layer9_ws_net(x, w, b):     
    x9 = w[8].T @ x + b[8]
    return x9
def z3_layer9_ws_net(x, w, b):     
    x9 = w[8].T @ x + b[8]
    return x9

def z3_layer1_ws_net_german(x, w, b):
    fl_x = np.array([Real('fl_x%s' % i) for i in range(20)])
    for i in range(len(x)):
        fl_x[i] = ToReal(x[i])        
    x1 = w[0].T @ fl_x + b[0]
    return x1

def z3_layer1_ws_net_bank(x, w, b):
    fl_x = np.array([Real('fl_x%s' % i) for i in range(16)])
    for i in range(len(x)):
        fl_x[i] = ToReal(x[i])        
    x1 = w[0].T @ fl_x + b[0]
    return x1

def harsh_prune(df, weight, bias, simulation_size, layer_net, range_dict):
    # sim_data = sim_df.drop(labels = [label_name], axis=1, inplace=False)
    x_data = df.drop(df.columns[len(df.columns)-1], axis=1, inplace=False)    
    sim_df = simluate_data(x_data, simulation_size, range_dict)

    candidates, _ = candidate_dead_nodes(sim_df.to_numpy(), weight, bias, layer_net)
    print(candidates)
    
    
    pr_w, pr_b = prune_neurons(weight, bias, candidates)
    compression = compression_ratio(candidates)
    print(round(compression, 2), '% pruning')
    
    return pr_w, pr_b

## Gives the lower and upper bound of both the WS layers and PL layers
def neuron_bounds(df, w, b, range_dict):
#    ub = df.max().values
#    lb = df.min().values   
    ub = []
    lb = []
    cols = df.columns.values
    for col_index in range(len(cols)):
        col = cols[col_index]
        lb.append(range_dict[col][0])
        ub.append(range_dict[col][1])
    
    lb_ws = []
    ub_ws = []
    
    lb_pl = []
    ub_pl = []
    
    for l in range(len(w)):
        in_size = len(w[l])
        layer_size = len(w[l][0])
        
        min_arr_ws = []
        max_arr_ws = []
        
        min_arr_pl = []
        max_arr_pl = []
    
        for j in range(layer_size):
            min_t = 0
            max_t = 0
    
            for i in range(in_size):
                weight = w[l][i][j]
    
                if weight < 0:
                    min_t += weight * ub[i]
                    max_t += weight * lb[i]
                else:
                    min_t += weight * lb[i]
                    max_t += weight * ub[i]
            
            min_ws = min_t + b[l][j]
            max_ws = max_t + b[l][j]
            min_arr_ws.append(min_ws)
            max_arr_ws.append(max_ws)
            
            min_pl = 0 if min_ws < 0 else min_ws
            max_pl = 0 if max_ws < 0 else max_ws
            min_arr_pl.append(min_pl)
            max_arr_pl.append(max_pl)
        
        lb = min_arr_pl
        ub = max_arr_pl
            
        lb_ws.append(min_arr_ws)
        ub_ws.append(max_arr_ws)
        
        lb_pl.append(min_arr_pl)
        ub_pl.append(max_arr_pl)
    return lb_ws, ub_ws, lb_pl, ub_pl
            

## Returns a mask of neurons where True => its a candidate for dead, False => otherwise
def candidate_dead_nodes(data, weight, bias, layer_net):
    layers = layer_net(data[0], weight, bias)
    layer_counts = []
    for l in layers:
        layer_counts.append([0] * len(l))
    
    # Count the number of non-zero occurence for each neuron
    for i in range(len(data)):
        layers = layer_net(data[i], weight, bias)
        for l in range(len(layers)):
            for j in range(len(layers[l])):
                if layers[l][j] != 0:
                    layer_counts[l][j] += 1
    #print(layer_counts)
    # Create mask for candidate dead neurons
    dead_nodes = copy.deepcopy(layer_counts) #layer_counts.copy()
    for l in dead_nodes:
        for i in range(len(l)):
            #l[i] = True if l[i] == 0 else False
            l[i] = 1 if l[i] == 0 else 0  # 1 means true
    positive_prob = copy.deepcopy(layer_counts) #layer_counts.copy()
    for l in positive_prob:
        for i in range(len(l)):
            l[i] = l[i]/len(data)
    return dead_nodes, positive_prob

def compression_ratio(deads):
    orig_neuron_count =  0
    compressed_neuron_count =  0
    for layer_index in range(len(deads)):
        for dead in deads[layer_index]:
            orig_neuron_count += 1
            if not dead:
                compressed_neuron_count += 1
                
    return 1 - (compressed_neuron_count/orig_neuron_count)

def simluate_data(df, size, range_dict):   
    cols = df.columns.values
    sim_data = []
    
    #print(range_dict)
    
    for i in range(size):    
        data_instance = []
        for col_index in range(len(cols)):
            col = cols[col_index]
            #print(col)
            data_instance.append(np.random.randint(range_dict[col][0], range_dict[col][1]+1))
        sim_data.append(data_instance)

    sim_data_arr = np.asarray(sim_data) # dtype=np.float32    
    sim_df = pd.DataFrame(data=sim_data_arr, columns=cols)
    
    return sim_df

# find neurons that has UB <=0 
# These are definitely dead
def dead_node_from_bound(cand, weight, bias, range_dict, ws_ub):
    print('INTERVAL BASED PRUNING')
    candidates = copy.deepcopy(cand) #candidates.copy()
    dead_node_mask = copy.deepcopy(candidates) #candidates.copy()
    #ws_lb, ws_ub, pl_lb, pl_ub = neuron_bounds(df, weight, bias, range_dict)
    count_finds = 0
    total_counts = 0
    
    for layer_index in range(len(ws_ub)):
        if(layer_index == len(ws_ub)-1):
            break
        
        for neuron_index in range(len(ws_ub[layer_index])):
            total_counts += 1
            
            if candidates[layer_index][neuron_index] == 0:
                continue
            
            if ws_ub[layer_index][neuron_index] <= 0:
                count_finds += 1
                #print('INACTIVE')
                dead_node_mask[layer_index][neuron_index] = 1 ## Adding dead nodes, 1=>dead
                candidates[layer_index][neuron_index] = 0 ## Not candidate anymore
            else:
                dead_node_mask[layer_index][neuron_index] = 0
    return dead_node_mask, candidates, count_finds/total_counts

def input_domain_constraint(df, x, ranges):
    props = []   
    cols = df.columns.values
    
    for col_index in range(len(cols)):
        col = cols[col_index]
        lb = ranges[col][0]
        ub = ranges[col][1]
        #print(lb, ub)
        props.append(And(x[col_index] >= lb, x[col_index] <= ub))        
    return props


def intermediate_domain_constraint(x, pl_lb, pl_ub, layer_index):
    props = []
    
    for neuron_index in range(len(pl_lb[layer_index])):
        #print(pl_lb[layer_index][neuron_index], pl_ub[layer_index][neuron_index])
        props.append(x[neuron_index] >= pl_lb[layer_index][neuron_index])
        props.append(x[neuron_index] <= pl_ub[layer_index][neuron_index])
    return props


def singular_verification(cand, df, weight, bias, ranges, pl_lb, pl_ub):
    print('SINGULAR VERIFICATION')
    # ws_lb, ws_ub, pl_lb, pl_ub = neuron_bounds(df, weight, bias, ranges)
    candidates = copy.deepcopy(cand)
    dead_node_mask = copy.deepcopy(candidates)      
    count_finds = 0
    total_counts = 0
    
    layer_index = 0
    
    for layer_index in range(len(candidates)):
        if(layer_index == len(candidates)-1):
            break
        #print('Layer ', layer_index)
        for neuron_index in range(len(candidates[layer_index])):
            #print('Neuron ', neuron_index)
            total_counts += 1
            
            if candidates[layer_index][neuron_index] == 0:
                continue
            
            if(layer_index == 0):
                x = np.array([Int('x%s' % i) for i in range(len(weight[layer_index]))])
                in_props = input_domain_constraint(df, x, ranges)
                y = z3_layer1_ws_net(x, weight, bias)

            elif(layer_index == 1):
                x = np.array([Real('x%s' % i) for i in range(len(weight[layer_index]))])
                in_props = intermediate_domain_constraint(x, pl_lb, pl_ub, layer_index)
                y = z3_layer2_ws_net(x, weight, bias)
                
            elif(layer_index == 2):
                x = np.array([Real('x%s' % i) for i in range(len(weight[layer_index]))])
                in_props = intermediate_domain_constraint(x, pl_lb, pl_ub, layer_index)
                y = z3_layer3_ws_net(x, weight, bias)

            elif(layer_index == 3):
                x = np.array([Real('x%s' % i) for i in range(len(weight[layer_index]))])
                in_props = intermediate_domain_constraint(x, pl_lb, pl_ub, layer_index)
                y = z3_layer4_ws_net(x, weight, bias)

            elif(layer_index == 4):
                x = np.array([Real('x%s' % i) for i in range(len(weight[layer_index]))])
                in_props = intermediate_domain_constraint(x, pl_lb, pl_ub, layer_index)
                y = z3_layer5_ws_net(x, weight, bias)
            
            elif(layer_index == 5):
                x = np.array([Real('x%s' % i) for i in range(len(weight[layer_index]))])
                in_props = intermediate_domain_constraint(x, pl_lb, pl_ub, layer_index)
                y = z3_layer6_ws_net(x, weight, bias)
            
            elif(layer_index == 6):
                x = np.array([Real('x%s' % i) for i in range(len(weight[layer_index]))])
                in_props = intermediate_domain_constraint(x, pl_lb, pl_ub, layer_index)
                y = z3_layer7_ws_net(x, weight, bias)
                
            elif(layer_index == 7):
                x = np.array([Real('x%s' % i) for i in range(len(weight[layer_index]))])
                in_props = intermediate_domain_constraint(x, pl_lb, pl_ub, layer_index)
                y = z3_layer8_ws_net(x, weight, bias)
            
            elif(layer_index == 8):
                x = np.array([Real('x%s' % i) for i in range(len(weight[layer_index]))])
                in_props = intermediate_domain_constraint(x, pl_lb, pl_ub, layer_index)
                y = z3_layer9_ws_net(x, weight, bias)

            s = Solver()             
            
            for i in in_props:    
                s.add(i)
            s.add(y[neuron_index] > 0)
            res = s.check()
            if res == unsat:
                #print('INACTIVE')
                count_finds += 1
                
                dead_node_mask[layer_index][neuron_index] = 1 ## Adding dead nodes, 1=>dead
                candidates[layer_index][neuron_index] = 0 ## Not candidate anymore
            else:
                dead_node_mask[layer_index][neuron_index] = 0

#            s.reset()
#            for i in in_props:    
#                s.add(i)
#            s.add(y[neuron_index] < 0)
#            if res == unsat:
#                print('ACTIVE: ', neuron_index)
                            
    return dead_node_mask, candidates, count_finds/total_counts

def singular_verification_german(cand, df, weight, bias, ranges, pl_lb, pl_ub):
    print('SINGULAR VERIFICATION')
    # ws_lb, ws_ub, pl_lb, pl_ub = neuron_bounds(df, weight, bias, ranges)
    candidates = copy.deepcopy(cand)
    dead_node_mask = copy.deepcopy(candidates)      
    count_finds = 0
    total_counts = 0
    
    layer_index = 0
    
    for layer_index in range(len(candidates)):
        if(layer_index == len(candidates)-1):
            break
        #print('Layer ', layer_index)
        for neuron_index in range(len(candidates[layer_index])):
            #print('Neuron ', neuron_index)
            total_counts += 1
            
            if candidates[layer_index][neuron_index] == 0:
                continue
            
            if(layer_index == 0):
                x = np.array([Int('x%s' % i) for i in range(len(weight[layer_index]))])
                in_props = input_domain_constraint(df, x, ranges)
                y = z3_layer1_ws_net_german(x, weight, bias)

            elif(layer_index == 1):
                x = np.array([Real('x%s' % i) for i in range(len(weight[layer_index]))])
                in_props = intermediate_domain_constraint(x, pl_lb, pl_ub, layer_index)
                y = z3_layer2_ws_net(x, weight, bias)
                
            elif(layer_index == 2):
                x = np.array([Real('x%s' % i) for i in range(len(weight[layer_index]))])
                in_props = intermediate_domain_constraint(x, pl_lb, pl_ub, layer_index)
                y = z3_layer3_ws_net(x, weight, bias)

            elif(layer_index == 3):
                x = np.array([Real('x%s' % i) for i in range(len(weight[layer_index]))])
                in_props = intermediate_domain_constraint(x, pl_lb, pl_ub, layer_index)
                y = z3_layer4_ws_net(x, weight, bias)

            elif(layer_index == 4):
                x = np.array([Real('x%s' % i) for i in range(len(weight[layer_index]))])
                in_props = intermediate_domain_constraint(x, pl_lb, pl_ub, layer_index)
                y = z3_layer5_ws_net(x, weight, bias)

            s = Solver()             
            
            for i in in_props:    
                s.add(i)
            s.add(y[neuron_index] > 0)
            res = s.check()
            if res == unsat:
                #print('INACTIVE')
                count_finds += 1
                
                dead_node_mask[layer_index][neuron_index] = 1 ## Adding dead nodes, 1=>dead
                candidates[layer_index][neuron_index] = 0 ## Not candidate anymore
            else:
                dead_node_mask[layer_index][neuron_index] = 0

#            s.reset()
#            for i in in_props:    
#                s.add(i)
#            s.add(y[neuron_index] < 0)
#            if res == unsat:
#                print('ACTIVE: ', neuron_index)
                            
    return dead_node_mask, candidates, count_finds/total_counts

def singular_verification_bank(cand, df, weight, bias, ranges, pl_lb, pl_ub):
    print('SINGULAR VERIFICATION')
    # ws_lb, ws_ub, pl_lb, pl_ub = neuron_bounds(df, weight, bias, ranges)
    candidates = copy.deepcopy(cand)
    dead_node_mask = copy.deepcopy(candidates)      
    count_finds = 0
    total_counts = 0
    
    layer_index = 0
    
    for layer_index in range(len(candidates)):
        if(layer_index == len(candidates)-1):
            break
        #print('Layer ', layer_index)
        for neuron_index in range(len(candidates[layer_index])):
            #print('Neuron ', neuron_index)
            total_counts += 1
            
            if candidates[layer_index][neuron_index] == 0:
                continue
            
            if(layer_index == 0):
                x = np.array([Int('x%s' % i) for i in range(len(weight[layer_index]))])
                in_props = input_domain_constraint(df, x, ranges)
                y = z3_layer1_ws_net_bank(x, weight, bias)

            elif(layer_index == 1):
                x = np.array([Real('x%s' % i) for i in range(len(weight[layer_index]))])
                in_props = intermediate_domain_constraint(x, pl_lb, pl_ub, layer_index)
                y = z3_layer2_ws_net(x, weight, bias)
                
            elif(layer_index == 2):
                x = np.array([Real('x%s' % i) for i in range(len(weight[layer_index]))])
                in_props = intermediate_domain_constraint(x, pl_lb, pl_ub, layer_index)
                y = z3_layer3_ws_net(x, weight, bias)

            elif(layer_index == 3):
                x = np.array([Real('x%s' % i) for i in range(len(weight[layer_index]))])
                in_props = intermediate_domain_constraint(x, pl_lb, pl_ub, layer_index)
                y = z3_layer4_ws_net(x, weight, bias)

            elif(layer_index == 4):
                x = np.array([Real('x%s' % i) for i in range(len(weight[layer_index]))])
                in_props = intermediate_domain_constraint(x, pl_lb, pl_ub, layer_index)
                y = z3_layer5_ws_net(x, weight, bias)

            s = Solver()             
            
            for i in in_props:    
                s.add(i)
            s.add(y[neuron_index] > 0)
            res = s.check()
            if res == unsat:
                #print('INACTIVE')
                count_finds += 1
                
                dead_node_mask[layer_index][neuron_index] = 1 ## Adding dead nodes, 1=>dead
                candidates[layer_index][neuron_index] = 0 ## Not candidate anymore
            else:
                dead_node_mask[layer_index][neuron_index] = 0

#            s.reset()
#            for i in in_props:    
#                s.add(i)
#            s.add(y[neuron_index] < 0)
#            if res == unsat:
#                print('ACTIVE: ', neuron_index)
                            
    return dead_node_mask, candidates, count_finds/total_counts

def sound_prune_global(df, weight, bias, simulation_size, layer_net, range_dict):
    x_df = df.drop(df.columns[len(df.columns)-1], axis=1, inplace=False)    
    sim_df = simluate_data(x_df, simulation_size, range_dict)

    candidates, _ = candidate_dead_nodes(sim_df.to_numpy(), weight, bias, layer_net)

    # Pruning based on the bounds
    ws_lb, ws_ub, pl_lb, pl_ub = neuron_bounds(sim_df, weight, bias, range_dict)
    
    b_dead_node_mask, b_candidates, b_compression = \
        dead_node_from_bound(candidates, weight, bias, range_dict, ws_ub)
    #print('Compr >>> ', b_compression)
    
    
    ## Tightenning the bound
    s_dead_node_mask, s_candidates, s_compression = \
        singular_verification(b_candidates, x_df, weight, bias, range_dict, pl_lb, pl_ub)
    #print('Compr >>> ', s_compression)
    dead_nodes = merge_dead_nodes(b_dead_node_mask, s_dead_node_mask)
    pr_w, pr_b = prune_neurons(weight, bias, dead_nodes)
    
    return pr_w, pr_b



def sound_prune(df, weight, bias, simulation_size, layer_net, range_dict):
    label_name = 'income-per-year'
    #sim_data = sim_df.drop(labels = [label_name], axis=1, inplace=False)
    x_df = df.drop(labels = [label_name], axis=1, inplace=False)
    sim_df = simluate_data(x_df, simulation_size, range_dict)

    candidates, pos_prob = candidate_dead_nodes(sim_df.to_numpy(), weight, bias, layer_net)
#    print('candi >>> ', candidates)

    # Pruning based on the bounds
    ws_lb, ws_ub, pl_lb, pl_ub = neuron_bounds(sim_df, weight, bias, range_dict)
    bounds = (ws_lb, ws_ub, pl_lb, pl_ub)
    
    #print(ws_lb)
    #print(ws_ub)
    
    b_dead_node_mask, b_candidates, b_compression = \
        dead_node_from_bound(candidates, weight, bias, range_dict, ws_ub)
    for l in b_dead_node_mask:
        if not 0 in l:
            l[0] = 0
            
    ## Tightenning the bound
    # --------------
    ## Use verification on WS nodes, layer by layer, check always +/- and merge/remove
    s_dead_node_mask, s_candidates, s_compression = \
        singular_verification(b_candidates, x_df, weight, bias, range_dict, pl_lb, pl_ub)
    for l in s_dead_node_mask:
        if not 0 in l:
            l[0] = 0
    
    dead_nodes = merge_dead_nodes(b_dead_node_mask, s_dead_node_mask)
    for l in dead_nodes:
        if not 0 in l:
            l[0] = 0
  
    return bounds, candidates, s_candidates, b_dead_node_mask, s_dead_node_mask, dead_nodes, pos_prob, sim_df

def sound_prune_bank(df, weight, bias, simulation_size, layer_net, range_dict):
    label_name = 'y'
    #sim_data = sim_df.drop(labels = [label_name], axis=1, inplace=False)
    x_df = df.drop(labels = [label_name], axis=1, inplace=False)
    sim_df = simluate_data(x_df, simulation_size, range_dict)

    candidates, pos_prob = candidate_dead_nodes(sim_df.to_numpy(), weight, bias, layer_net)
#    print('candi >>> ', candidates)

    # Pruning based on the bounds
    ws_lb, ws_ub, pl_lb, pl_ub = neuron_bounds(sim_df, weight, bias, range_dict)
    bounds = (ws_lb, ws_ub, pl_lb, pl_ub)
    
    #print(ws_lb)
    #print(ws_ub)
    
    b_dead_node_mask, b_candidates, b_compression = \
        dead_node_from_bound(candidates, weight, bias, range_dict, ws_ub)
    for l in b_dead_node_mask:
        if not 0 in l:
            l[0] = 0
            
    ## Tightenning the bound
    # --------------
    ## Use verification on WS nodes, layer by layer, check always +/- and merge/remove
    s_dead_node_mask, s_candidates, s_compression = \
        singular_verification_bank(b_candidates, x_df, weight, bias, range_dict, pl_lb, pl_ub)
    for l in s_dead_node_mask:
        if not 0 in l:
            l[0] = 0
    
    dead_nodes = merge_dead_nodes(b_dead_node_mask, s_dead_node_mask)
    for l in dead_nodes:
        if not 0 in l:
            l[0] = 0
  
    return bounds, candidates, s_candidates, b_dead_node_mask, s_dead_node_mask, dead_nodes, pos_prob, sim_df

def sound_prune_german(df, weight, bias, simulation_size, layer_net, range_dict):
    label_name = 'credit'
    #sim_data = sim_df.drop(labels = [label_name], axis=1, inplace=False)
    x_df = df.drop(labels = [label_name], axis=1, inplace=False)
    sim_df = simluate_data(x_df, simulation_size, range_dict)

    candidates, pos_prob = candidate_dead_nodes(sim_df.to_numpy(), weight, bias, layer_net)
#    print('candi >>> ', candidates)

    # Pruning based on the bounds
    ws_lb, ws_ub, pl_lb, pl_ub = neuron_bounds(sim_df, weight, bias, range_dict)
    bounds = (ws_lb, ws_ub, pl_lb, pl_ub)
    
    #print(ws_lb)
    #print(ws_ub)
    
    b_dead_node_mask, b_candidates, b_compression = \
        dead_node_from_bound(candidates, weight, bias, range_dict, ws_ub)
    for l in b_dead_node_mask:
        if not 0 in l:
            l[0] = 0
            
    ## Tightenning the bound
    # --------------
    ## Use verification on WS nodes, layer by layer, check always +/- and merge/remove
    s_dead_node_mask, s_candidates, s_compression = \
        singular_verification_german(b_candidates, x_df, weight, bias, range_dict, pl_lb, pl_ub)
    for l in s_dead_node_mask:
        if not 0 in l:
            l[0] = 0
    
    dead_nodes = merge_dead_nodes(b_dead_node_mask, s_dead_node_mask)
    for l in dead_nodes:
        if not 0 in l:
            l[0] = 0
  
    return bounds, candidates, s_candidates, b_dead_node_mask, s_dead_node_mask, dead_nodes, pos_prob, sim_df


def heuristic_prune(bounds, candidates, s_candidates, deads, pos_prob, perc_threshold, w, b):
    (ws_lb, ws_ub, pl_lb, pl_ub) = bounds
    
    new_deads = copy.deepcopy(candidates)
    for l in new_deads:
        for i in range(len(l)):
            l[i] = 0 # Refresh the neurons
    
    for layer_id in range(len(candidates)):
        if layer_id == len(candidates)-1:
            break
        cand = []
        noncand = []
        for neuron_id in range(len(candidates[layer_id])):

            if candidates[layer_id][neuron_id]:
                cand.append(ws_ub[layer_id][neuron_id])
                
            else:
                noncand.append(ws_ub[layer_id][neuron_id])
                
        if len(noncand) == 0:
            for neuron_id in range(len(s_candidates[layer_id])):
                new_deads[layer_id][neuron_id] = 1
        elif len(cand) == 0:
            pass
        else:
        
            cand_min = min(cand)
            noncand_min = min(noncand)
            
            cand_max = max(cand)
            noncand_max = max(noncand)
            
            cand_mean = np.mean(np.array(cand))
            noncand_mean = np.mean(np.array(noncand))
            
            cand_median = np.median(np.array(cand))
            noncand_median = np.median(np.array(noncand))
            
            cand_5perc = np.percentile(np.array(cand), perc_threshold)
            noncand_5perc = np.percentile(np.array(noncand), perc_threshold)
            
            cand_95perc = np.percentile(np.array(cand), 100-perc_threshold)
            noncand_95perc = np.percentile(np.array(noncand), 100-perc_threshold)
            
            if noncand_mean > 2*cand_mean and noncand_median > 2*cand_median:
                ## now, we made sure that distributions are different
                
                for neuron_id in range(len(s_candidates[layer_id])):
                # filtrations
                    if s_candidates[layer_id][neuron_id]:
                        
                        #print(neuron_id, ws_ub[layer_id][neuron_id])
                        if ws_ub[layer_id][neuron_id] < noncand_5perc:
                            if ws_ub[layer_id][neuron_id] < 0.1 * noncand_95perc: # most harsh
                                
                                if ws_ub[layer_id][neuron_id] < abs(ws_lb[layer_id][neuron_id]):
                                    
                                    new_deads[layer_id][neuron_id] = 1
                                    #print(neuron_id)
                            
          
        
        #print(cand_min, cand_max, cand_mean, cand_median, cand_5perc, cand_95perc)
        #print(noncand_min, noncand_max, noncand_mean, noncand_median, noncand_5perc, noncand_95perc)
    
    for l in new_deads:
        if not 0 in l:
            l[0] = 0
             
    merged_deads = merge_dead_nodes(deads, new_deads)
    for l in merged_deads:
        if not 0 in l:
            l[0] = 0
       
    
    return new_deads, merged_deads

def merge_dead_nodes(b, s):
    merged = copy.deepcopy(b)

    for layer_index in range(len(s)):
        for neuron_index in range(len(s[layer_index])):
            if s[layer_index][neuron_index] == 1:
                merged[layer_index][neuron_index] = 1        
    return merged
    
def prune_neurons(weight, bias, candidates):
    ## Actual pruning
    pr_w = copy.deepcopy(weight)
    pr_b = copy.deepcopy(bias)
    
#    print('++++++++')
    
    for i in range(len(weight)): # layer i
#        print(np.shape(weight[i]))
        c = 0
        for j in range(len(candidates[i])):
          
            if candidates[i][j]:
                
                #if c != len(candidates[i]):
                pr_w[i] = np.delete(pr_w[i], j-c, 1)
                pr_b[i] = np.delete(pr_b[i], j-c, 0)
                if(i != len(weight)-1):
                    pr_w[i+1] = np.delete(pr_w[i+1], j-c, 0)
                c += 1
                
#    print('--------')
#    for i in range(len(pr_w)): # layer i
#        print(np.shape(pr_w[i]))           

#    print(pr_w[0].shape)
    print('Pruning done!')
    return pr_w, pr_b

