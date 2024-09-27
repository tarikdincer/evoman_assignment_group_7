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

from evoman.environment import Environment
from demo_controller import player_controller

# imports other libs
import numpy as np
from scipy.special import softmax
from scipy.stats import percentileofscore
import os
import matplotlib.pyplot as plt

# runs simulation
def simulation(env,x):
    f,p,e,t = env.play(pcont=x)
    return f

# evaluation
def evaluate(env, x):
    return np.array(list(map(lambda y: simulation(env,y), x)))


run_mode = 'train'
experiment_type = "dynamic"
headless = True
if headless:
    os.environ["SDL_VIDEODRIVER"] = "dummy"


experiment_name = 'optimization_task1'
if not os.path.exists(experiment_name):
    os.makedirs(experiment_name)

n_hidden_neurons = 10

# initializes simulation in individual evolution mode, for single static enemy.
env = Environment(experiment_name=experiment_name,
                enemies=[3],
                playermode="ai",
                player_controller=player_controller(n_hidden_neurons), # you  can insert your own controller here
                enemymode="static",
                level=2,
                speed="fastest",
                visuals=False)


# number of weights for multilayer with 10 hidden neurons
n_vars = (env.get_num_sensors()+1)*n_hidden_neurons + (n_hidden_neurons+1)*5

ratio = 0.33
dom_u = 1
dom_l = -1
npop = 50
gens = 25
mutation = 0.2
num_step_size = 1
learning_rate = 1/sqrt(n_vars)
init_step_size = 0.05
u_boundary_step_size = 1
l_boundary_step_size = 0
l_percentile_guided = 25
u_percentile_guided = 75
guided_influence = 0.1

# start writing your own code from here
def survival_selection(pop, fit_pop, num_selected):
    min_fitness = np.min(fit_pop)
    # # shifting the fitness values to make sure it contains no negative
    # if min_fitness < 0:
    #     fit_pop = fit_pop - min_fitness
    normalized_fitness = fit_pop
    if min_fitness < 0:
        normalized_fitness = fit_pop - min_fitness
        
    
    probs = normalized_fitness / np.sum(normalized_fitness)

    probs = np.maximum(probs, 0.01)
    probs = probs / np.sum(probs)
    
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
def boundaries_step_size(x):
    if x>u_boundary_step_size:
        return u_boundary_step_size
    elif x<l_boundary_step_size:
        return l_boundary_step_size
    else:
        return x

# limits
def limits(x):

    if x>dom_u:
        return dom_u
    elif x<dom_l:
        return dom_l
    else:
        return x
    
    
# calculates the percintile of the of the offspring's fitness in the population 
def fitness_percentile(fitness, fit_pop):
    offspring_fitness_percentile = percentileofscore(fit_pop, fitness)

    return offspring_fitness_percentile

# weighted average crossover function
def crossover(pop, fit_pop, parents):
    print("Crossover with dynamic mutation")
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
            new_step_size = boundaries_step_size(new_step_size)

            # set the new stepsize for the offspring
            offspring[f][n_vars] = new_step_size

            # mutation using updated step size
            for i in range(0,len(offspring[f] - num_step_size)):
                if np.random.uniform(0 ,1)<=mutation:
                    offspring[f][i] =   offspring[f][i]+np.random.normal(0, new_step_size)

            offspring[f] = np.array(list(map(lambda y: limits(y), offspring[f])))

            total_offspring = np.vstack((total_offspring, offspring[f]))

    return total_offspring

# weighted average crossover function with static mutation
def crossover_static_mutation(pop, fit_pop, parents):
    # initialize output array 
    total_offspring = np.zeros((0,n_vars + 1))
    
    for p in range(0,len(parents), 2):

        # select parents
        p1 = pop[parents[p]]
        p2 = pop[parents[p+1]]
        
        n_offspring = 1
        offspring =  np.zeros((n_offspring, n_vars + num_step_size))

        # mutation
        for f in range(1):
            # get weights for weighted average by using softmax
            w1, w2 = softmax([fit_pop[parents[p]], fit_pop[parents[p+1]]])
            # cross_prob = np.random.uniform(0,1)
            offspring[f] = w1 * p1 + p2 * w2


            # mutation using updated step size
            for i in range(0,len(offspring[f] - num_step_size)):
                if np.random.uniform(0 ,1)<=mutation:
                    offspring[f][i] =   offspring[f][i]+np.random.normal(0, init_step_size)

            offspring[f] = np.array(list(map(lambda y: limits(y), offspring[f])))

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

def test_solution(alg_idx, run_idx):
    bsol = np.loadtxt(f'{experiment_name}/best{alg_idx}_{run_idx}.txt')
    print( '\n RUNNING SAVED BEST SOLUTION \n')
    env.update_parameter('speed','normal')
    x = evaluate([bsol])

    return x

def plot_fitness(static_mean_avg, static_best_avg,
                 dynamic_mean_avg, dynamic_best_avg, enemy):
    
    generations = np.arange(1, gens + 1)
    
    plt.figure(figsize=(10, 6))
    
    plt.plot(generations, static_mean_avg, label='Static Mutation - Mean Fitness', 
                  color='red', linestyle='-')
    plt.plot(generations, static_best_avg, label='Static Mutation - Best Fitness', 
                  color='red', linestyle='--')
    plt.plot(generations, dynamic_mean_avg, label='Dynamic Mutation - Mean Fitness', 
                  color='blue', linestyle='-')
    plt.plot(generations, dynamic_best_avg, label='Dynamic Mutation - Best Fitness', 
                  color='blue', linestyle='--')

    plt.title(f'Fitness Progression for Static and Dynamic Mutation for Enemy {enemy}')
    plt.xlabel('Generations')
    plt.ylabel('Fitness')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_std(static_mean_std, static_best_std,
                 dynamic_mean_std, dynamic_best_std, enemy):
    
    generations = np.arange(1, gens + 1)
    
    plt.figure(figsize=(10, 6))
    
    plt.plot(generations, static_mean_std, label='Static Mutation - Mean Std', 
                  color='red', linestyle='-')
    plt.plot(generations, static_best_std, label='Static Mutation - Best Std', 
                  color='red', linestyle='--')
    plt.plot(generations, dynamic_mean_std, label='Dynamic Mutation - Mean Std', 
                  color='blue', linestyle='-')
    plt.plot(generations, dynamic_best_std, label='Dynamic Mutation - Best Std', 
                  color='blue', linestyle='--')

    plt.title(f'Standart Deviation Progression for Static and Dynamic Mutation for Enemy {enemy}')
    plt.xlabel('Generations')
    plt.ylabel('Standart Deviation')
    plt.legend()
    plt.grid(True)
    plt.show()

def calculate_avg_std(stats, n_runs, n_gens):
    mean_fit_avg = np.zeros(n_gens)
    mean_fit_std = np.zeros(n_gens)
    best_fit_avg = np.zeros(n_gens)
    best_fit_std = np.zeros(n_gens)

    for gen in range(n_gens):
        mean_fit_gen = [stats[run][gen]['mean'] for run in range(n_runs)]
        best_fit_gen = [stats[run][gen]['best_fit'] for run in range(n_runs)]

        mean_fit_avg[gen] = np.mean(mean_fit_gen)
        mean_fit_std[gen] = np.std(mean_fit_gen)
        best_fit_avg[gen] = np.mean(best_fit_gen)
        best_fit_std[gen] = np.std(best_fit_gen)

    return mean_fit_avg, mean_fit_std, best_fit_avg, best_fit_std

def run_evolution(n_runs):
    static_stats = []
    dynamic_stats = []
    enemy_set = [3, 5, 8]
    for en in enemy_set:
        env.update_parameter('enemies',[en])
        for n_alg in range(2):
            experiment_type = "static" if n_alg == 0 else "dynamic"
            for n_run in range(n_runs):
                print(f'Running {n_run + 1}. run with {experiment_type} mutation')
                print( '\nNEW EVOLUTION\n')
                
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
                file_aux  = open(f'{experiment_name}/results_{n_alg+1}_{n_run+1}.txt','a')
                file_aux.write('\n\ngen best mean std')
                print( '\n GENERATION '+str(ini_g)+' '+str(round(fit_pop[best],6))+' '+str(round(mean,6))+' '+str(round(std,6)))
                file_aux.write('\n'+str(ini_g)+' '+str(round(fit_pop[best],6))+' '+str(round(mean,6))+' '+str(round(std,6))   )
                file_aux.close()

                if experiment_type == "static":
                    static_stats.append([])  # Initialize list for this run
                else:
                    dynamic_stats.append([])  # Initialize list for this run

                # evolution

                for i in range(ini_g+1, gens + 1):
                    ini = round(time.time() * 1000)
                    if experiment_type == "static":
                        static_stats.append([])
                    else:
                        dynamic_stats.append([])
                    
                    parents = truncation_random_hybrid_selection(pop, fit_pop, pop.shape[0] // 2, ratio)
                    random.shuffle(parents)
                    offspring = crossover(pop, fit_pop, parents) if experiment_type == "dynamic" else crossover_static_mutation(pop, fit_pop, parents) # crossover
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
                    end = round(time.time() * 1000)

                    generation_stats = {"mean": mean, "best": best, "std": std, "best_fit": fit_pop[best], "best": pop[best], "time": round((end-ini))}
                    
                    if experiment_type == 'static':
                        static_stats[n_run].append(generation_stats)
                    else:
                        dynamic_stats[n_run].append(generation_stats)


                    # saves results
                    file_aux  = open(f'{experiment_name}/results_{n_alg+1}_{n_run+1}.txt','a')
                    print( '\n GENERATION '+str(i)+' '+str(round(fit_pop[best],6))+' '+str(round(mean,6))+' '+str(round(std,6)))
                    file_aux.write('\n'+str(i)+' '+str(round(fit_pop[best],6))+' '+str(round(mean,6))+' '+str(round(std,6))   )
                    file_aux.close()

                    # saves generation number
                    file_aux  = open(f'{experiment_name}/gen_{n_alg+1}_{n_run+1}.txt','w')
                    file_aux.write(str(i))
                    file_aux.close()

                    # saves file with the best solution
                    np.savetxt(f'{experiment_name}/best_{n_alg+1}_{n_run+1}.txt',pop[best])

                    # saves simulation state
                    solutions = [pop, fit_pop]
                    env.update_solutions(solutions)
                    env.save_state()
                
        for idx, stat in enumerate(static_stats):
            for gen_idx, gen in enumerate(stat):
                print(f"Static {idx + 1}. run {gen_idx + 1}. generation results: Mean:{stat[gen_idx]["mean"]} Std:{stat[gen_idx]["std"]} Best Fitness:{stat[gen_idx]["best_fit"]} Time:{stat[gen_idx]["time"]}")
        for idx, stat in enumerate(dynamic_stats):
            for gen_idx, gen in enumerate(stat):
                print(f"Dynamic {idx + 1}. run {gen_idx + 1}. generation results: Mean:{stat[gen_idx]["mean"]} Std:{stat[gen_idx]["std"]} Best Fitness:{stat[gen_idx]["best_fit"]} Time:{stat[gen_idx]["time"]}")
        
        static_mean_avg, static_mean_std, static_best_avg, static_best_std = calculate_avg_std(static_stats, n_runs, gens)
        dynamic_mean_avg, dynamic_mean_std, dynamic_best_avg, dynamic_best_std = calculate_avg_std(dynamic_stats, n_runs, gens)    

        plot_fitness(static_mean_avg, static_best_avg,
             dynamic_mean_avg, dynamic_best_avg, en)
        plot_std(static_mean_std, static_best_std,
             dynamic_mean_std, dynamic_best_std, en)
    

run_evolution(10)