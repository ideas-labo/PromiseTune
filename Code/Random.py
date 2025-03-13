import numpy as np
import random
import warnings
from util import get_objective
from util import read_file

warnings.filterwarnings('ignore')

def random_sample(filename, maxlives, budget, seed):
    random.seed(seed)
    np.random.seed(seed)
    initial_size = 0
    lives = maxlives
    steps = 0
    file = read_file.get_data(filename, initial_size, seed)
    training_dep = [t.objective[-1] for t in file.training_set]
    training_indep = [t.decision for t in file.training_set]
    result = float('inf')
    for x in training_dep:
        if result > x:
            result = x
    results, training_dep, x_axis, xs = [], [], [], []
    exist_configuration = training_indep[:]
    while steps < budget:
        tmp=[]
        for index in range(len(file.independent_set)):
            tmp.append(random.choice(file.independent_set[index]))
        reward,configuration = get_objective.get_objective(file.dict_search,tmp)
        training_indep.append(tmp)
        training_dep.append(reward)
        x_axis.append(steps)
        results.append(reward)
        xs.append(configuration)
        if configuration in exist_configuration:
            budget += 1
        else:
            exist_configuration.append(configuration)
        if reward < result:
            result = reward
            lives = maxlives
        else:
            lives -= 1
        if lives == 0:
            break
        steps += 1
        print('Loop:', steps, ' Reward:', '{:.2f}'.format(reward), ' Config:', configuration)
    return np.array(xs), np.array(results), result

def run_main(seed, name):
    xs, result, res = random_sample(
        filename=f"./Data/{name}.csv",
        maxlives=200,
        budget=100,
        seed=seed,
    )
    print( xs, result, res)

if __name__ == '__main__':
    seeds = [1]
    # mp.freeze_support()
    # pool = mp.Pool(processes=200)
    systems = ['exastencils','dconvert','7z',"BDBC",'deeparch',
                 'PostgreSQL','javagc','storm','x264',
                 'redis', 'HSQLDB','LLVM']
    for name in systems:
        for seed in seeds:
            run_main(seed, name)
    # pool.close()
    # pool.join()
    
    
