import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import random
import copy
import lingam
from sklearn.ensemble import RandomForestRegressor
from scipy import stats
from itertools import product
from causallearn.search.ConstraintBased.FCI import fci
from util import get_objective, read_file
from util.matrix import get_adjacency_matrix
from util.helper import get_tree_paths, has_negative_value, in_rule, Model, merge_constraints
from util.ei import get_ei

def cause_find(training_indep,training_dep,file,sample):
    global sample_plus
    feature_names = file.features
    all_path = []
    model = RandomForestRegressor(n_estimators=5, min_samples_leaf=int(sample))
    model.fit(training_indep, training_dep)
    sub_trees = model.estimators_
    for _, sub_tree in enumerate(sub_trees):
        paths = get_tree_paths(sub_tree, feature_names)
        for path in paths:
            if path[:] not in all_path:
                all_path.append(path[:])
    new_rows, new_perfs = [], []
    for row,perf in zip(training_indep, training_dep):
        if row not in new_rows: 
            new_rows.append(row)
            new_perfs.append(perf)
    if len(all_path) >= len(new_rows)-2:
        sample_plus += 1
        all_path = random.sample(all_path, len(new_rows)-2)
    try:
        ACEs, valid_rules = model_fit(new_rows, all_path, feature_names, new_perfs)
    except Exception as e:
        all_path = random.sample(all_path, len(all_path)-2)
        ACEs, valid_rules = model_fit(new_rows, all_path, feature_names, new_perfs)
    return ACEs,valid_rules
    
def model_fit(new_rows,all_path,feature_names,new_perfs):
    column_names = copy.deepcopy(feature_names)
    training_indep_new = copy.deepcopy(new_rows)
    for i,x in enumerate(new_rows):
        for j,rule in enumerate(all_path):
            if in_rule(x,rule,feature_names):
                training_indep_new[i].append(1)
            else:
                training_indep_new[i].append(0)
        training_indep_new[i].append(new_perfs[i])
    labels = []
    for a in range(len(all_path)):
        labels.append("rule"+str(a))
        column_names.append("rule"+str(a))
    labels.append("perf")
    column_names.append("perf")
    df = pd.DataFrame(training_indep_new, columns=column_names)
    df.drop(columns=feature_names, inplace=True)
    data = df.to_numpy()
    try:
        g, _ = fci(data, show_progress=False, verbose=False)
        adjacency_matrix = get_adjacency_matrix(g).T
        adjacency_matrix[:, -1] = 0
    except Exception as e:
        adjacency_matrix = np.zeros((len(labels), len(labels)), dtype=int)
    prior_knowledge = adjacency_matrix
    prior_knowledge[-1, :-1] = 1
    prior_knowledge[-1, -1] = -1
    model = lingam.DirectLiNGAM(prior_knowledge=prior_knowledge)
    model.fit(data)
    valid_rules, ACEs = [], []
    for k in range(len(labels)-1):
        ACE = model.estimate_total_effect(data, k, len(labels)-1)
        if ACE:
            valid_rules.append(all_path[k])
            ACEs.append(ACE)
    return ACEs,valid_rules

def random_search_with_static_distribution(max_iterations, stop_threshold, y, estimators, eta, every_column):
    best_value = float('-inf')
    values, xs = [], []
    for i in range(max_iterations):
        pred = []
        current_x = list(random.choice(y))
        for index in range(len(current_x)):
            if current_x[index] == '_':
                current_x[index] = random.choice(every_column[index])
        for e in estimators:
            pred.append(e.predict([current_x]))
        current_value = get_ei(pred, eta)[0]
        xs.append(current_x)
        values.append(current_value)
        if current_value > best_value:
            best_value = current_value
        if i > 0:
            try:
                kde = stats.gaussian_kde(values)
            except Exception:
                break
            improvement_probability = 1 - kde.integrate_box_1d(-np.inf, best_value)
            if improvement_probability < stop_threshold:
                break
    return xs, best_value

def random_search_with_static_distribution_cold(max_iterations, stop_threshold, y, estimators, eta):
    best_value = float('-inf')
    values, xs = [], []
    for i in range(max_iterations):
        pred = []
        current_x = []
        for index in range(len(y)):
            current_x.append(random.choice(y[index]))
        for e in estimators:
            pred.append(e.predict([current_x]))
        current_value = get_ei(pred, eta)[0]
        values.append(current_value)
        xs.append(current_x)
        if current_value > best_value:
            best_value = current_value
        if i > 0:
            try:
                kde = stats.gaussian_kde(values)
            except Exception:
                break
            improvement_probability = 1 - kde.integrate_box_1d(-np.inf, best_value)
            if improvement_probability < stop_threshold:
                break  
    return xs, best_value

def get_training_sequence_by_PromiseTune(training_indep, training_dep, eta, file, rule, sample, k):
    global sample_plus
    p = 0.01
    max_iter = int(1e4)
    model = Model(training_indep, training_dep)
    best_configs = sorted(zip(training_indep, training_dep), key=lambda x: x[1])[:int(k*len(training_indep))]
    estimators = model.estimators_
    every_column = file.independent_set
    header = file.features
    Rule, candidates = [], []
    if rule == True:
        ACEs, all_path = cause_find(training_indep, training_dep, file, sample+sample_plus)
        for i, path in zip(ACEs, all_path):
            if i<0:
                for indep, _ in best_configs:
                    if in_rule(indep, path, header):
                        Rule += path
                        print("==>RULE:", [merge_constraints(path)], " ==>ACE:",i)
                        break
    if (rule == True and not has_negative_value(ACEs)) or rule == False:
        xs, _ = random_search_with_static_distribution_cold(max_iter, p, every_column, estimators, eta)
        candidates.extend(xs)
    else:
        for index in range(len(all_path[:])):
            if ACEs[index]>0:
                continue
            results = [[] for _ in range(len(header))]
            single_rule = all_path[index]
            for i, column_values, single_feature in zip(range(len(header)), every_column, header):
                tmp_left, tmp_right = [], []
                for single_component in single_rule:
                    if single_component[0] == single_feature:
                        if single_component[2] == 'L':
                            tmp_left.append(single_component[1])
                        else:
                            tmp_right.append(single_component[1])
                if tmp_left and tmp_right: 
                    for x in column_values:
                        if max(tmp_right)<=x<=min(tmp_left):
                            results[i].append(x)
                elif tmp_left:
                    for x in column_values:
                        if x<=min(tmp_left):
                            results[i].append(x)
                elif tmp_right:
                    for x in column_values:
                        if max(tmp_right)<=x:
                            results[i].append(x)
            for i in range(len(results)):
                if not results[i]:
                    results[i] = ['_']
                elif len(results[i]) > 10:
                    results[i] = random.sample(results[i], 10)
            new_list = list(product(*results))
            xs, _ = random_search_with_static_distribution(max_iter, p, new_list, estimators, eta, every_column)
            candidates.extend(xs)
    return candidates, model, Rule

def get_best_configuration_id_PromiseTune(training_indep, training_dep, eta, file, rule, sample, k):
    test_sequence, model, Rule = get_training_sequence_by_PromiseTune(training_indep, training_dep, eta, file, rule, sample, k)
    estimators = model.estimators_
    pred = []
    for e in estimators:
        pred.append(e.predict(test_sequence))
    value = get_ei(pred, eta)
    merged_predicted_objectives = [[i,a] for a, i in zip(value, test_sequence)]
    merged_predicted_objectives = sorted(merged_predicted_objectives, key=lambda x: x[1], reverse=True)
    for x in merged_predicted_objectives:
        if x[0] not in training_indep:
            return x[0], Rule

def PromiseTune(filename, initial_size, maxlives, budget, seed, rule, l, k):
    global sample_plus
    random.seed(seed)
    np.random.seed(seed)
    steps, sample_plus = 0, 0
    configuration_b = None
    results, x_axis, xs, Rules = [], [], [], []
    result = float('inf')
    lives = maxlives
    file = read_file.get_data(filename, initial_size,seed)
    training_indep = [t.decision for t in file.training_set]
    training_dep = [t.objective[-1] for t in file.training_set]
    for x in training_dep:
        if result > x:
            result = x
    training_dep = []
    for action in training_indep:
        reward,configuration = get_objective.get_objective(file.dict_search, action)
        training_dep.append(reward)
        xs.append(action)
        results.append(reward)    

    exist_configuration = training_indep[:]
    while initial_size + steps < budget:
        steps += 1
        best_solution, Rule = get_best_configuration_id_PromiseTune(
            training_indep, training_dep, result, file, rule, l, k)
        best_result, configuration = get_objective.get_objective(
            file.dict_search, best_solution)
        Rules.append(Rule)
        training_indep.append(best_solution)
        training_dep.append(best_result)
        x_axis.append(steps)
        xs.append(configuration)
        results.append(best_result)
        if best_result < result:
            configuration_b = configuration
            result = best_result
            lives = maxlives
        else:
            lives -= 1
        if configuration in exist_configuration:
            budget += 1
        else:
            exist_configuration.append(configuration)
        print('=>Loop:', steps, ' =>Reward:', '{:.2f}'.format(best_result), ' =>Config:', configuration)
        if lives == 0:
            break
    return configuration_b, Rules

def run_main(seed, name):
    best_configuration, Rules = PromiseTune(
        filename=f"./Data/{name}.csv",
        initial_size=10,
        maxlives=200,
        budget=100,
        seed=seed,
        rule=True,
        l=10,
        k=0.1
    )
    print(best_configuration, Rules)

if __name__ == '__main__':
    seeds = [1]
    # mp.freeze_support()
    # pool = mp.Pool(processes=200)

    for name in ['exastencils','dconvert','7z',"BDBC",'deeparch',
                 'PostgreSQL','javagc','storm','x264',
                 'redis', 'HSQLDB','LLVM']:
        for seed in seeds:
            run_main(seed, name)
    # pool.close()
    # pool.join()
    