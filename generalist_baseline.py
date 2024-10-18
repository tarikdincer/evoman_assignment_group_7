###############################################################################
# EvoMan FrameWork - V1.0 2016  			                                  #
# DEMO : Neuroevolution - Genetic Algorithm  neural network.                  #
# Author: Karine Miras        			                                      #
# karine.smiras@gmail.com     				                                  #
###############################################################################

# imports framework
import sys
import time
from math import fabs,sqrt,exp
import random
#import optuna

from evoman.environment import Environment
from demo_controller import player_controller

# imports other libs
import numpy as np
from scipy.special import softmax
from scipy.stats import percentileofscore
import os

# runs simulation
def simulation(env,x):
    f,p,e,t = env.play(pcont=x)
    return f

# evaluation
def evaluate(env, x):
    return np.array(list(map(lambda y: simulation(env,y), x)))

ini = time.time()  # sets time marker
run_mode = 'train'
experiment_type = "dynamic"
headless = True
if headless:
    os.environ["SDL_VIDEODRIVER"] = "dummy"


experiment_name = 'generalist_baseline'
if not os.path.exists(experiment_name):
    os.makedirs(experiment_name)

n_hidden_neurons = 10

num_step_size = 1

# start writing your own code from here
def survival_selection(pop, fit_pop, num_selected):
    min_fitness = np.min(fit_pop)
    # # shifting the fitness values to make sure it contains no negative
    # if min_fitness < 0:
    #     fit_pop = fit_pop - min_fitness
    normalized_fitness = fit_pop
    if min_fitness < 0:
        normalized_fitness = fit_pop - min_fitness + 0.0000001
        
    
    probs = normalized_fitness / np.sum(normalized_fitness)
    
    best_idx = np.argmax(fit_pop)
    best_nn = pop[best_idx]
    best_fit = fit_pop[best_idx]

    num_selected = num_selected - 1
    
    if num_selected > (pop.shape[0] - 1):
        num_selected = (pop.shape[0] - 1)

    indices = np.random.choice(pop.shape[0], num_selected, p = probs, replace=False)

    selected_pop = pop[indices]
    selected_fit = fit_pop[indices]

    selected_pop = np.vstack([selected_pop, best_nn])
    selected_fit = np.append(selected_fit, best_fit)
    
    return selected_pop, selected_fit

# ensures that the step_size stays within these boundaries
"""def boundaries_step_size(x):
    if x>u_boundary_step_size:
        return u_boundary_step_size
    elif x<l_boundary_step_size:
        return l_boundary_step_size
    else:
        return x"""

# limits
def limits(x, l, u):

    if x>u:
        return u
    elif x<l:
        return l
    else:
        return x
    
    
# calculates the percintile of the of the offspring's fitness in the population 
def fitness_percentile(fitness, fit_pop):
    offspring_fitness_percentile = percentileofscore(fit_pop, fitness)

    return offspring_fitness_percentile

# weighted average crossover function
def crossover(pop, fit_pop, parents, learning_rate, guided_influence, mutation, l_boundary_step_size, u_boundary_step_size, dom_l, dom_u, l_percentile_guided, u_percentile_guided, n_vars, env):
    # initialize output array 
    total_offspring = np.zeros((0,n_vars + 1))
    
    for p in range(0,len(parents), 2):

        # select parents
        p1 = pop[parents[p]]
        p2 = pop[parents[p+1]]
        
        n_offspring = 1
        offspring =  np.zeros((n_offspring, n_vars + num_step_size))

        # TODO: add mutation
        for f in range(1):
            # get weights for weighted average by using softmax
            w1, w2 = softmax([fit_pop[parents[p]], fit_pop[parents[p+1]]])
            # cross_prob = np.random.uniform(0,1)
            offspring[f] = w1 * p1 + p2 * w2

            #check if initializing the stepsize randomly affects the performance of the model

            # calculate the fitness of the offspring and compute it's percentile in the population
            offspring_fitness = simulation(env, offspring[f])
            percentile = fitness_percentile(offspring_fitness, fit_pop)

            # obtain the current step_size from the offspring 
            step_size = offspring[f][n_vars]
            
            new_step_size = 0

            # update the stepsize for adaptive step-size but also take the fitness of the offspring into account by adjusting the mean of the normal distribution
            # if the offspring is below a certain percentile in the population mean is increased such that the stepsize on average slightly increases and conversely
            # if the offspring is above a certain percentile decrease the mean slightly to decrease the stepsize
            if percentile < l_percentile_guided:
                new_step_size = step_size * exp(learning_rate * np.random.normal(guided_influence, 1))
            elif percentile > u_percentile_guided:
                new_step_size = step_size * exp(learning_rate * np.random.normal( - guided_influence, 1))
            else:
                new_step_size = step_size * exp(learning_rate * np.random.normal(0, 1))

            # ensures step size stays within boundaries
            new_step_size = limits(new_step_size, l_boundary_step_size, u_boundary_step_size)

            # set the new stepsize for the offspring
            offspring[f][n_vars] = new_step_size

            # mutation using updated step size
            for i in range(0,(len(offspring[f]) - num_step_size)):
                if np.random.uniform(0 ,1)<=mutation:
                    offspring[f][i] =   offspring[f][i]+np.random.normal(0, new_step_size)

            offspring[f] = np.array(list(map(lambda y: limits(y, dom_l, dom_u), offspring[f])))

            total_offspring = np.vstack((total_offspring, offspring[f]))

    return total_offspring

# weighted average crossover function with static mutation
def crossover_static_mutation(pop, fit_pop, parents, mutation, init_step_size, dom_l, dom_u, n_vars):
    # initialize output array 
    total_offspring = np.zeros((0,n_vars + 1))
    
    for p in range(0,len(parents), 2):

        # select parents
        p1 = pop[parents[p]]
        p2 = pop[parents[p+1]]
        
        n_offspring = 1
        offspring =  np.zeros((n_offspring, n_vars + num_step_size))

        # TODO: add mutation
        for f in range(1):
            # get weights for weighted average by using softmax
            w1, w2 = softmax([fit_pop[parents[p]], fit_pop[parents[p+1]]])
            # cross_prob = np.random.uniform(0,1)
            offspring[f] = w1 * p1 + p2 * w2


            # mutation using updated step size
            for i in range(0,len(offspring[f]) - num_step_size):
                if np.random.uniform(0 ,1)<=mutation:
                    offspring[f][i] =   offspring[f][i]+np.random.normal(0, init_step_size)

            offspring[f] = np.array(list(map(lambda y: limits(y, dom_l, dom_u), offspring[f])))

            total_offspring = np.vstack((total_offspring, offspring[f]))

    return total_offspring

def truncation_random_hybrid_selection(pop,fit, nparents, ratio):
# 
# Returns a list of parents to be used in the crossover. The indices to the parents are provided.
# Uses a mix of Truncation Selection and random selection to generate a list of parents.
# Through truncation selection a certain number of best individuals are selected as parents.
# The parent list is then filled up by randomly sampling from the remaining individuals.
# pop: list that contains the population which will be sampled
# fit: list that contains the corresponding fitness values of the population
# nparents: the amount of parents to be generated
# ratio: the ratio between the parents chosen through truncation and random selection.
# For example a ratio of 3 will yield 1/3 selected through truncation and 2/3 selected randomly.
# 
    popsize = len(pop)
    nparents = nparents + 1 if nparents % 2 == 1 else nparents
    indices = list(range(popsize))
    ratio = int(1//ratio)
    slicesize = nparents//ratio
    combined = list(zip(fit,indices))

    sorted_combined = sorted(combined, key = lambda x:x[0], reverse=True) #highest first
    sorted_population = [ind for _, ind in sorted_combined]
    best_individuals = sorted_population[:slicesize]
    other_ind = sorted_population[slicesize:]
    random_ind = list(np.random.choice(other_ind, nparents - slicesize, False))

    return best_individuals + random_ind

def pop_std(pop):
    return np.mean(np.std(np.stack(pop), axis=0))


# loads file with the best solution for testing
"""if run_mode =='test':

    bsol = np.loadtxt(experiment_name+'/best.txt')
    print( '\n RUNNING SAVED BEST SOLUTION \n')
    env.update_parameter('speed','normal')
    evaluate([bsol])

    sys.exit(0)"""

evolution_number = 0

def evolutionary_algorithm(evo_param):
    # Extract the EA parameters from the dictionary initialized in the optuna objective function
    for run in range(10):
        ratio = evo_param['ratio']
        dom_u = evo_param['dom_u']
        dom_l = evo_param['dom_l']
        npop = evo_param['npop']
        gens = 5
        mutation = evo_param['mutation']
        learning_rate = evo_param['learning_rate']
        init_step_size = evo_param['init_step_size']
        u_boundary_step_size = 1
        l_boundary_step_size = 0
        l_percentile_guided = evo_param['l_percentile_guided']
        u_percentile_guided = evo_param['u_percentile_guided']
        guided_influence = evo_param['guided_influence']

        # global evolution_number
        # personal_number = evolution_number
        # # to keep track on the number of evolutions optuna has done
        # evolution_number += 1

        env = Environment(experiment_name=experiment_name,
                    enemies=[3,5,7,8],
                    multiplemode="yes",
                    playermode="ai",
                    player_controller=player_controller(n_hidden_neurons), # you  can insert your own controller here
                    enemymode="static",
                    level=2,
                    speed="fastest",
                    visuals=False)


        # number of weights for multilayer with 10 hidden neurons
        n_vars = (env.get_num_sensors()+1)*n_hidden_neurons + (n_hidden_neurons+1)*5

        print( '\nNEW EVOLUTION ' + str(run) + '\n')
        print('parameters for current evolution: ')
        for key, value in evo_param.items():
            print(f'{key}: {value}')

        pop = np.random.uniform(dom_l, dom_u, (npop, n_vars)) # creating npop size nn's with weights in between -1 and 1
        pop = np.hstack([pop, np.full((pop.shape[0], 1), init_step_size)])
        fit_pop = evaluate(env, pop) #returns an array that stores the fitness of each nn
        best = np.argmax(fit_pop)
        mean = np.mean(fit_pop)
        std = np.std(fit_pop)
        ini_g = 0
        solutions = [pop, fit_pop]
        env.update_solutions(solutions)
            

        # saves results for first pop
        file_aux  = open(experiment_name + '/results' + str(run) + '.txt', 'a')
        file_aux.write('\n\ngen best mean std div')
        file_aux.write('\n'+str(ini_g)+' '+str(round(fit_pop[best],6))+' '+str(round(mean,6))+' '+str(round(std,6)) + ' ' + str(round(pop_std(pop), 6))  )
        file_aux.close()
        print( '\n GENERATION '+str(ini_g)+' '+str(round(fit_pop[best],6))+' '+str(round(mean,6))+' '+str(round(std,6)) + ' ' + str(round(pop_std(pop), 6)))

        # evolution

        last_sol = fit_pop[best]
        notimproved = 0

        for i in range(ini_g+1, gens):
            parents = truncation_random_hybrid_selection(pop, fit_pop, pop.shape[0] // 2, ratio)
            random.shuffle(parents)
            offspring = crossover(pop, fit_pop, parents, learning_rate, guided_influence, mutation, l_boundary_step_size, u_boundary_step_size, dom_l, dom_u, l_percentile_guided, u_percentile_guided, n_vars, env) if experiment_type == "dynamic" else crossover_static_mutation(pop, fit_pop, parents, mutation, init_step_size, dom_l, dom_u, n_vars) # crossover
            fit_offspring = evaluate(env, offspring)   # evaluation
            pop = np.vstack((pop,offspring))
            fit_pop = np.append(fit_pop,fit_offspring)

            best = np.argmax(fit_pop) #best solution in generation
            fit_pop[best] = float(evaluate(env, np.array([pop[best] ]))[0]) # repeats best eval, for stability issues
            best_sol = fit_pop[best]

            pop, fit_pop = survival_selection(pop, fit_pop, npop)

            best = np.argmax(fit_pop)
            std  =  np.std(fit_pop)
            mean = np.mean(fit_pop)


            # saves results
            file_aux  = open(experiment_name + '/results' + str(run) + '.txt', 'a')
            file_aux.write('\n'+str(i)+' '+str(round(fit_pop[best],6))+' '+str(round(mean,6))+' '+str(round(std,6)) + ' ' + str(round(pop_std(pop), 6)))
            file_aux.close()
            print( '\n GENERATION '+str(i)+' from evolution '+ str(run) + ' ' + str(round(fit_pop[best],6)) + ' ' + str(round(mean,6)) + ' ' + str(round(std,6)) + ' ' + str(round(pop_std(pop),6)))

            # saves generation number
            file_aux  = open(experiment_name+'/gen.txt','w')
            file_aux.write(str(i))
            file_aux.close()

            # saves file with the best solution
            np.savetxt(experiment_name + '/best' + str(run) + '.txt' , pop[best])

            # saves simulation state
            solutions = [pop, fit_pop]
            env.update_solutions(solutions)
            env.save_state()

        #return fit_pop[best]

# Optuna objective function
def objective(trial):
    evo_param = {
        'ratio': trial.suggest_float('ratio', 0.25, 0.75),
        'dom_u': trial.suggest_float('dom_u', 0.75, 1.5),
        'dom_l': trial.suggest_float('dom_l', -1.5,-0.75),
        'npop': trial.suggest_int('npop', 100, 150, step=10),
        'gens': trial.suggest_int('gens', 20, 30, step=1),
        'mutation': trial.suggest_float('mutation', 0.1, 0.5, step=0.1),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2),
        'init_step_size': trial.suggest_float('init_step_size', 0.05, 0.5),
        'u_boundary_step_size': 1,
        'l_boundary_step_size': 0,
        'l_percentile_guided': trial.suggest_int('l_percentile_guided', 5, 40, step=5),
        'u_percentile_guided': trial.suggest_int('u_percentile_guided', 60, 95, step=5),
        'guided_influence': trial.suggest_float('guided_influence', 0.0, 0.4)
        }

    best = evolutionary_algorithm(evo_param)

    return best

# study = optuna.create_study(direction='maximize')
# study.optimize(objective, n_trials=50, n_jobs=-1)
# params = study.best_trial.params
# print('Best trial:', study.best_trial.params)

evoparams = {'ratio': 0.40888884079383736, 'dom_u': 1.3288762810281702, 'dom_l': -1.3323823224940499, 'npop': 140, 'mutation': 0.30000000000000004, 'learning_rate': 0.06308261258328778, 'init_step_size': 0.09946600396334193, 'l_percentile_guided': 15, 'u_percentile_guided': 90, 'guided_influence': 0.15822732383302338}

best = evolutionary_algorithm(evoparams)

fim = time.time() # prints total execution time for experiment
print( '\nExecution time: '+str(round((fim-ini)/60))+' minutes \n')
print( '\nExecution time: '+str(round((fim-ini)))+' seconds \n')




#env.state_to_log() # checks environment state