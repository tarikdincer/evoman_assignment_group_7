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


experiment_name = 'optimization_test_task'
if not os.path.exists(experiment_name):
    os.makedirs(experiment_name)

n_hidden_neurons = 10

# For testing 1 generalist island and 2 specialist islands. Will be changed later
# initializes simulation in individual evolution mode, for single static enemy.
envgeneral = Environment(experiment_name=experiment_name,
                enemies=[3,5,8],
                multiplemode="yes",
                playermode="ai",
                player_controller=player_controller(n_hidden_neurons), # you  can insert your own controller here
                enemymode="static",
                level=2,
                speed="fastest",
                visuals=False)

env1 = Environment(experiment_name=experiment_name,
                enemies=[1],
                playermode="ai",
                player_controller=player_controller(n_hidden_neurons), # you  can insert your own controller here
                enemymode="static",
                level=2,
                speed="fastest",
                visuals=False)

env2 = Environment(experiment_name=experiment_name,
                enemies=[7],
                playermode="ai",
                player_controller=player_controller(n_hidden_neurons), # you  can insert your own controller here
                enemymode="static",
                level=2,
                speed="fastest",
                visuals=False)


# number of weights for multilayer with 10 hidden neurons
n_vars = (envgeneral.get_num_sensors()+1)*n_hidden_neurons + (n_hidden_neurons+1)*5

ratio = 0.33
dom_u = 1
dom_l = -1
npop = 100
gens = 100
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
def survival_selection(pop, fit_pop, num_selected, epoch=False):
    min_fitness = np.min(fit_pop)
    # # shifting the fitness values to make sure it contains no negative
    # if min_fitness < 0:
    #     fit_pop = fit_pop - min_fitness
    normalized_fitness = fit_pop
    if min_fitness < 0:
        normalized_fitness = fit_pop - min_fitness
        
    
    probs = normalized_fitness / np.sum(normalized_fitness)
    
    best_idx = np.argmax(fit_pop)
    best_nn = pop[best_idx]
    best_fit = fit_pop[best_idx]

    num_selected = num_selected - 1
    
    if num_selected > (pop.shape[0] - 1):
        num_selected = (pop.shape[0] - 1)

    if epoch:
        indices = np.random.choice(pop.shape[0], num_selected, replace=False) # Random for now, i think the other one might be unfavourable for the migrated individuals. I might be wrong though.
    else:
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
            offspring_fitness = simulation(envgeneral, offspring[f])
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
    print("Crossover with static mutation")
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
            for i in range(0,len(offspring[f] - num_step_size)):
                if np.random.uniform(0 ,1)<=mutation:
                    offspring[f][i] =   offspring[f][i]+np.random.normal(0, init_step_size)

            offspring[f] = np.array(list(map(lambda y: limits(y), offspring[f])))

            total_offspring = np.vstack((total_offspring, offspring[f]))

    return total_offspring

# weighted average crossover function
def crossover_epoch(poplist, fit_poplist, parentslist):
    print("Epoch Crossover with dynamic mutation")
    # initialize output array 
    total_offspring = np.zeros((0,n_vars + 1))
    
    for p in range(0,len(parentslist[0])):

        # select parents
        p1 = poplist[0][parentslist[0][p]]
        p2 = poplist[1][parentslist[1][p]]
        p3 = poplist[2][parentslist[2][p]]
        
        n_offspring = 1
        offspring =  np.zeros((n_offspring, n_vars + num_step_size))

        # TODO: add mutation
        for f in range(1):
            # get weights for weighted average by using softmax
            w1, w2, w3 = softmax([fit_poplist[0][parentslist[0][p]], fit_poplist[1][parentslist[1][p]], fit_poplist[2][parentslist[2][p]]])
            # cross_prob = np.random.uniform(0,1)
            offspring[f] = w1 * p1 + p2 * w2 + p3 * w3

            #check if initializing the stepsize randomly affects the performance of the model

            # calculate the fitness of the offspring and compute it's percentile in the population
            offspring_fitness = simulation(envgeneral, offspring[f])
            percentile = fitness_percentile(offspring_fitness, fit_poplist[0])

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

def crossover_static_mutation_epoch(poplist, fit_poplist, parentslist):
    print("Epoch Crossover with static mutation")
    # initialize output array 
    total_offspring = np.zeros((0,n_vars + 1))
    
    for p in range(0,len(parentslist[0])):

        # select parents
        p1 = poplist[0][parentslist[0][p]]
        p2 = poplist[1][parentslist[1][p]]
        p3 = poplist[2][parentslist[2][p]]
        
        n_offspring = 1
        offspring =  np.zeros((n_offspring, n_vars + num_step_size))

        # TODO: add mutation
        for f in range(1):
            # get weights for weighted average by using softmax
            w1, w2, w3 = softmax([fit_poplist[0][parentslist[0][p]], fit_poplist[1][parentslist[1][p]], fit_poplist[2][parentslist[2][p]]])
            # cross_prob = np.random.uniform(0,1)
            offspring[f] = w1 * p1 + p2 * w2 + p3 * w3


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


# loads file with the best solution for testing
if run_mode =='test':

    bsol = np.loadtxt(experiment_name+'/best.txt')
    print( '\n RUNNING SAVED BEST SOLUTION \n')
    envgeneral.update_parameter('speed','normal')
    evaluate([bsol])

    sys.exit(0)


# initializes population loading old solutions or generating new ones

if not os.path.exists(experiment_name+'/evoman_solstate'):

    print( '\nNEW EVOLUTION\n')
    print("Init generalist population\n")

    popgeneral = np.random.uniform(dom_l, dom_u, (npop, n_vars)) # creating npop size nn's with weights in between -1 and 1
    popgeneral = np.hstack([popgeneral, np.full((popgeneral.shape[0], 1), init_step_size)])
    fit_popgeneral = evaluate(envgeneral, popgeneral) #returns an array that stores the fitness of each nn
    bestgeneral = np.argmax(fit_popgeneral)
    meangeneral = np.mean(fit_popgeneral)
    stdgeneral = np.std(fit_popgeneral)
    ini_g = 0
    solutionsgeneral = [popgeneral, fit_popgeneral]
    envgeneral.update_solutions(solutionsgeneral)
    
    print("Init specialist population 1\n")

    pop1 = np.random.uniform(dom_l, dom_u, (npop, n_vars)) # creating npop size nn's with weights in between -1 and 1
    pop1 = np.hstack([pop1, np.full((pop1.shape[0], 1), init_step_size)])
    fit_pop1 = evaluate(env1, pop1) #returns an array that stores the fitness of each nn
    best1 = np.argmax(fit_pop1)
    mean1 = np.mean(fit_pop1)
    std1 = np.std(fit_pop1)
    # ini_g = 0
    solutions1 = [pop1, fit_pop1]
    env1.update_solutions(solutions1)

    print("Init specialist population 2\n")
    pop2 = np.random.uniform(dom_l, dom_u, (npop, n_vars)) # creating npop size nn's with weights in between -1 and 1
    pop2 = np.hstack([pop2, np.full((pop2.shape[0], 1), init_step_size)])
    fit_pop2 = evaluate(env2, pop2) #returns an array that stores the fitness of each nn
    best2 = np.argmax(fit_pop2)
    mean2 = np.mean(fit_pop2)
    std2 = np.std(fit_pop2)
    # ini_g = 0
    solutions2 = [pop2, fit_pop2]
    env2.update_solutions(solutions2)
else:

    print( '\nCONTINUING EVOLUTION\n')

    envgeneral.load_state()
    popgeneral = envgeneral.solutions[0]
    fit_popgeneral = envgeneral.solutions[1]

    bestgeneral = np.argmax(fit_popgeneral)
    meangeneral = np.mean(fit_popgeneral)
    stdgeneral = np.std(fit_popgeneral)

    # finds last generation number
    file_aux  = open(experiment_name+'/gen.txt','r')
    ini_g = int(file_aux.readline())
    file_aux.close()




# saves results for first pop
file_aux  = open(experiment_name+'/results.txt','a')
file_aux.write('\n\ngen best mean std')
print( '\n GENERATION '+str(ini_g)+' '+str(round(fit_popgeneral[bestgeneral],6))+' '+str(round(meangeneral,6))+' '+str(round(stdgeneral,6)))
file_aux.write('\n'+str(ini_g)+' '+str(round(fit_popgeneral[bestgeneral],6))+' '+str(round(meangeneral,6))+' '+str(round(stdgeneral,6))   )
file_aux.close()


# evolution

last_sol = fit_popgeneral[bestgeneral]
notimproved = 0

def evolve_generation(pop, fit_pop, env):
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
    solutions = [popgeneral, fit_popgeneral]
    envgeneral.update_solutions(solutions)
    envgeneral.save_state()

    return pop, fit_pop

for i in range(ini_g+1, gens):
    popgeneral, fit_popgeneral = evolve_generation(popgeneral, fit_popgeneral, envgeneral)


    bestgeneral = np.argmax(fit_popgeneral)
    stdgeneral  =  np.std(fit_popgeneral)
    meangeneral = np.mean(fit_popgeneral)


    # saves results
    file_aux  = open(experiment_name+'/results.txt','a')
    print( '\n GENERATION '+str(i)+' '+str(round(fit_popgeneral[bestgeneral],6))+' '+str(round(meangeneral,6))+' '+str(round(stdgeneral,6)))
    file_aux.write('\n'+str(i)+' '+str(round(fit_popgeneral[bestgeneral],6))+' '+str(round(meangeneral,6))+' '+str(round(stdgeneral,6))   )
    file_aux.close()

    # saves generation number
    file_aux  = open(experiment_name+'/gen.txt','w')
    file_aux.write(str(i))
    file_aux.close()

    # saves file with the best solution
    np.savetxt(experiment_name+'/best.txt',popgeneral[bestgeneral])

    pop1, fit_pop1 = evolve_generation(pop1, fit_pop1, env1)

    best1 = np.argmax(fit_pop1)

    pop2, fit_pop2 = evolve_generation(pop2, fit_pop2, env2)

    best2 = np.argmax(fit_pop2)

    if i % 10 == 0:
        parentsgeneral = truncation_random_hybrid_selection(popgeneral, fit_popgeneral, 20, ratio) # 20 = placeholder, we will determine this amount later
        random.shuffle(parentsgeneral)
        parentsspec = truncation_random_hybrid_selection(pop1, fit_pop1, 20, ratio)
        random.shuffle(parentsspec)
        parentsspec2 = truncation_random_hybrid_selection(pop2, fit_pop2, 20, ratio)
        random.shuffle(parentsspec2)
        offspringepoch = crossover_epoch([popgeneral, pop1,pop2], [fit_popgeneral, fit_pop1, fit_pop2], 
                                         [parentsgeneral,parentsspec, parentsspec2]) if experiment_type == "dynamic" else crossover_static_mutation_epoch([popgeneral, pop1,pop2], 
                                                                                                                    [fit_popgeneral, fit_pop1, fit_pop2], [parentsgeneral,parentsspec, parentsspec2]) # can easily be expandeded to more islands
        fit_offspringepoch = evaluate(envgeneral, offspringepoch)
        popgeneral = np.vstack((popgeneral, offspringepoch))
        fit_popgeneral = np.append(fit_popgeneral, fit_offspringepoch)
        popgeneral, fit_popgeneral = survival_selection(popgeneral, fit_popgeneral, npop, True)
        solutionsgeneral = [popgeneral, fit_popgeneral]
        envgeneral.update_solutions(solutionsgeneral)
        envgeneral.save_state()




fim = time.time() # prints total execution time for experiment
print( '\nExecution time: '+str(round((fim-ini)/60))+' minutes \n')
print( '\nExecution time: '+str(round((fim-ini)))+' seconds \n')




envgeneral.state_to_log() # checks environment state

    