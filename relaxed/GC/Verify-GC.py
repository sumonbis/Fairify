#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
sys.path.append('../../')

from random import shuffle
from z3 import * 
from utils.input_partition import *
from utils.verif_utils import *
from utils.prune import *
from importlib import import_module

# In[]

df, X_train, y_train, X_test, y_test = load_german()
X = np.r_[X_train, X_test]
single_input = X_test[0].reshape(1, 20)
#print_metadata(df)

# In[]
model_dir = '../../models/german/'
result_dir = './res/sex-marital-'
PARTITION_THRESHOLD = 10

SOFT_TIMEOUT = 100 
HARD_TIMEOUT = 60*60
HEURISTIC_PRUNE_THRESHOLD = 20

# In[]
## Domain
default_range = [0, 1]
range_dict = {}
range_dict['status'] = [0, 2]
range_dict['month'] = [0, 80]
range_dict['credit_history'] = [0, 2]
range_dict['purpose'] = [0, 9]
range_dict['credit_amount'] = [0, 20000]
range_dict['savings'] = [0, 2]
range_dict['employment'] = [0, 2]
range_dict['investment_as_income_percentage'] = [1, 4]
range_dict['other_debtors'] = [0, 2]
range_dict['residence_since'] = [1, 4]
range_dict['property'] = [0, 2]
range_dict['age'] = [0, 1]
range_dict['installment_plans'] = [0, 2]
range_dict['housing'] = [0, 2]
range_dict['number_of_credits'] = [1, 4]
range_dict['skill_level'] = [0, 3]
range_dict['people_liable_for'] = [1, 2]
range_dict['telephone'] = [0, 1]
range_dict['foreign_worker'] = [0, 1]
range_dict['sex'] = [0, 1]

A = range_dict.keys()
PA = ['sex', 'marital-status']

RA = []
RA_threshold = 100

sim_size = 1 * 1000

p_dict = partition(range_dict, PARTITION_THRESHOLD)
p_list = partitioned_ranges(A, PA, p_dict, range_dict)
print('Number of partitions: ', len(p_list))
shuffle(p_list)

# In[]

model_files = os.listdir(model_dir)
for model_file in model_files:
    if not model_file.endswith('.h5'):
        continue;
    print('==================  STARTING MODEL ' + model_file)
    model_name = model_file.split('.')[0]
    
    model_funcs = 'utils.' + model_name + '-Model-Functions'
    mod = import_module(model_funcs)
    layer_net = getattr(mod, 'layer_net')
    net = getattr(mod, 'net')
    z3_net = getattr(mod, 'z3_net')

    w = []
    b = []
    
    model = load_model(model_dir + model_file)
    
    for i in range(len(model.layers)):
        w.append(model.layers[i].get_weights()[0])
        b.append(model.layers[i].get_weights()[1])
        
    print('###################')
    partition_id = 0
    sat_count = 0
    unsat_count = 0
    unk_count = 0
    cumulative_time = 0
    
    for p in p_list:
        heuristic_attempted = 0
        result = []
        start_time = time.time()
    
        partition_id += 1
        simulation_size = 1*1000
    
        
        # In[]
    #    sd = s
        neuron_bounds, candidates, s_candidates, b_deads, s_deads, st_deads, pos_prob, sim_X_df  = \
            sound_prune_german(df, w, b, simulation_size, layer_net, p)
    
        b_compression = compression_ratio(b_deads)
        s_compression = compression_ratio(s_deads)
        st_compression = compression_ratio(st_deads)
    
        pr_w, pr_b = prune_neurons(w, b, st_deads)
        
        
    
        # In[]
        # Create properties
        in_props = []
        out_props = []
    
        x = np.array([Int('x%s' % i) for i in range(20)]) 
        x_ = np.array([Int('x_%s' % i) for i in range(20)])
    
        y = z3_net(x, pr_w, pr_b) # y is an array of size 1
        y_ = z3_net(x_, pr_w, pr_b)
    
        # Basic fairness property - must include
        for attr in A:
            if(attr in PA):
                in_props.extend(in_const_german(df, x, attr, 'neq', x_))
            elif(attr in RA):
                in_props.extend(in_const_diff_german(df, x, x_, attr, RA_threshold))
            else:
                in_props.extend(in_const_german(df, x, attr, 'eq', x_))
    
        in_props.extend(in_const_domain_german(df, x, x_, p, PA))
    
        # In[]
        s = Solver()
        #s.reset()
    
        if(len(sys.argv) > 1):
            s.set("timeout", int(sys.argv[1]) * 1000) # X seconds
        else:
            s.set("timeout", SOFT_TIMEOUT * 1000)
    
    
        for i in in_props:
            s.add(i)
    
        s.add(Or(And(y[0] < 0, y_[0] > 0), And(y[0] > 0, y_[0] < 0)))
    
        print('Verifying ...')
        res = s.check()
    
        print(res)
        if res == sat:
            m = s.model()
            inp1, inp2 = parse_z3Model(m)
        
        sv_time = s.statistics().time
        s_end_time = time.time()
        s_time = compute_time(start_time, s_end_time)
        hv_time = 0
        # In[]
        h_compression = 0
        t_compression = st_compression
        h_success = 0
        if res == unknown:
            heuristic_attempted = 1
    
            h_deads, deads = heuristic_prune(neuron_bounds, candidates,
                s_candidates, st_deads, pos_prob, HEURISTIC_PRUNE_THRESHOLD, w, b)
    
            del pr_w
            del pr_b
    
            pr_w, pr_b = prune_neurons(w, b, deads)
            h_compression = compression_ratio(h_deads)
            print(round(h_compression*100, 2), '% HEURISTIC PRUNING')
            t_compression = compression_ratio(deads)
            print(round(t_compression*100, 2), '% TOTAL PRUNING')
    
            y = z3_net(x, pr_w, pr_b) # y is an array of size 1
            y_ = z3_net(x_, pr_w, pr_b)
    
            s = Solver()
    
            if(len(sys.argv) > 1):
                s.set("timeout", int(sys.argv[1]) * 1000) # X seconds
            else:
                s.set("timeout", SOFT_TIMEOUT * 1000)
    
            for i in in_props:
                s.add(i)
    
            s.add(Or(And(y[0] < 0, y_[0] > 0), And(y[0] > 0, y_[0] < 0)))
            print('Verifying ...')
            res = s.check()
    
            print(res)
            if res == sat:
                m = s.model()
                inp1, inp2 = parse_z3Model(m)
                
            if res != unknown:
                h_success = 1
            hv_time = s.statistics().time
    
        # In[]
        h_time = compute_time(s_end_time, time.time())
        total_time = compute_time(start_time, time.time())
    
        cumulative_time += total_time
    
        # In[]
        print('V time: ', s.statistics().time)
        file = result_dir + model_name + '.csv'
    
        # In[]
        c_check_correct = 0
        accurate = 0
        d1 = ''
        d2 = ''
        if res == sat:
            sat_count += 1
            d1 = np.asarray(inp1, dtype=np.float32)
            d2 = np.asarray(inp2, dtype=np.float32)
            print(inp1)
            print(inp2)
            res1 = net(d1, pr_w, pr_b)
            res2 = net(d2, pr_w, pr_b)
            print(res1, res2)
            pred1 = sigmoid(res1)
            pred2 = sigmoid(res2)
            class_1 = pred1 > 0.5
            class_2 = pred2 > 0.5
            
            res1_orig = net(d1, w, b)
            res2_orig = net(d2, w, b)
            print(res1_orig, res2_orig)
            pred1_orig = sigmoid(res1_orig)
            pred2_orig = sigmoid(res2_orig)
            class_1_orig = pred1_orig > 0.5
            class_2_orig = pred2_orig > 0.5
            
            if class_1_orig != class_2_orig:
                accurate = 1
            if class_1 == class_1_orig and class_2 == class_2_orig:
                c_check_correct = 1
        elif res == unsat:
            unsat_count += 1
        else:
            unk_count +=1
            
    
        d = X_test[0]
        res1 = net(d, pr_w, pr_b)
        pred1 = sigmoid(res1)
        class_1 = pred1 > 0.5
    
        res1_orig = net(d, w, b)
        pred1_orig = sigmoid(res1_orig)
        class_1_orig = pred1_orig > 0.5
        
        sim_X = sim_X_df.to_numpy()    
        sim_y_orig = get_y_pred(net, w, b, sim_X)    
        sim_y = get_y_pred(net, pr_w, pr_b, sim_X)
        
       
        orig_acc = accuracy_score(y_test, get_y_pred(net, w, b, X_test))
        pruned_acc = accuracy_score(sim_y_orig, sim_y)

        # In[]
        res_cols = ['Partition_ID', 'Verification', 'SAT_count', 'UNSAT_count', 'UNK_count', 'h_attempt', 'h_success', \
                    'B_compression', 'S_compression', 'ST_compression', 'H_compression', 'T_compression', 'SV-time', 'S-time', 'HV-Time', 'H-Time', 'Total-Time', 'C-check',\
                    'V-accurate', 'Original-acc', 'Pruned-acc', 'Acc-dec', 'C1', 'C2']
    
        result.append(partition_id)
        result.append(str(res))
        result.append(sat_count)
        result.append(unsat_count)
        result.append(unk_count)
        result.append(heuristic_attempted)
        result.append(h_success)
        result.append(round(b_compression, 4))
        result.append(round(s_compression, 4))
        result.append(round(st_compression, 4))
        result.append(round(h_compression, 4))
        result.append(round(t_compression, 4))
        result.append(sv_time)
        result.append(s_time)
        result.append(hv_time)
        result.append(h_time)
        result.append(total_time)
        result.append(c_check_correct)
        result.append(accurate)
        result.append(round(orig_acc, 4))
        result.append(round(pruned_acc, 4))
        result.append('-')
        #result.append(round(orig_acc - pruned_acc, 4))
        result.append(d1)
        result.append(d2)
    
    
        import csv
        file_exists = os.path.isfile(file)
        with open(file, "a", newline='') as fp:
            if not file_exists:
                wr = csv.writer(fp, dialect='excel')
                wr.writerow(res_cols)
    
            wr = csv.writer(fp)
            wr.writerow(result)
        print('******************')
        
        if(cumulative_time > HARD_TIMEOUT):
            print('==================  COMPLETED MODEL ' + model_file)
            break