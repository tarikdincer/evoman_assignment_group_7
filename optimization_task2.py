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


experiment_name = 'optimization_test_task2'
if not os.path.exists(experiment_name):
    os.makedirs(experiment_name)

n_hidden_neurons = 10

# For testing 1 generalist island and 2 specialist islands. Will be changed later
# initializes simulation in individual evolution mode, for single static enemy.

# Please keep this commented out, otherwise it just causes confusion
# env_general = Environment(experiment_name=experiment_name,
#                 enemies=[3,5,8],
#                 multiplemode="yes",
#                 playermode="ai",
#                 player_controller=player_controller(n_hidden_neurons), # you  can insert your own controller here
#                 enemymode="static",
#                 level=2,
#                 speed="fastest",
#                 visuals=False)

islands = [{'env': 'generalist', 'enemies': [3,5,7,8]},
         #{'env': 'generalist', 'enemies': [3,5,7,8]},
         {'env': 'specialist', 'enemies': [3]},
         {'env': 'specialist', 'enemies': [5]},
         {'env': 'specialist', 'enemies': [7]},
         {'env': 'specialist', 'enemies': [8]}]

# number of weights for multilayer with 10 hidden neurons
num_sensors = 20
n_vars = (num_sensors+1)*n_hidden_neurons + (n_hidden_neurons+1)*5

ratio = 0.33
dom_u = 1
dom_l = -1
npop = 100
gens = 200
mutation = 0.2
num_step_size = 1
learning_rate = 1/sqrt(n_vars)
init_step_size = 0.05
u_boundary_step_size = 1
l_boundary_step_size = 0
l_percentile_guided = 25
u_percentile_guided = 75
guided_influence = 0.1
similar_migration = True
epoch_number = 0 
epoch_frequency = 20
start_val_gen_ratio = 0.2
end_val_gen_ratio = 0.5

total_epochs = gens / epoch_frequency

# start writing your own code from here
def init_islands():
    env_gens = []
    env_specs = []
    for island in islands:
        env = Environment(experiment_name=experiment_name,
                enemies=island['enemies'],
                multiplemode="yes" if island['env'] == 'generalist' else "no",
                playermode="ai",
                player_controller=player_controller(n_hidden_neurons), # you  can insert your own controller here
                enemymode="static",
                level=2,
                speed="fastest",
                visuals=False)
        
        if island['env'] == 'generalist':
            env_gens.append(env)
        else:
            env_specs.append(env)
        
    return env_gens, env_specs

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
    # print(len(pop))
    # print(len(fit_pop))
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

def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

def is_most_similar_island(offspring_individual, islands, index):
    avg_similarities = [
        np.mean([cosine_similarity(offspring_individual[:n_vars], individual[:n_vars]) for individual in island])
        for island in islands
    ]
    # Check if the similarity of the island at the given index is the maximum
    return avg_similarities[index] == max(avg_similarities)

def similar_island(offspring_individual, islands, probability_threshold=0): # currently uses random offspring placement by default
    if random.random() >= probability_threshold:
        return random.randint(0, len(islands) - 1)

    avg_similarities = [
        np.mean([cosine_similarity(offspring_individual[:n_vars], individual[:n_vars]) for individual in island])
        for island in islands
    ]
    
    most_similar_index = np.argmax(avg_similarities)
    return most_similar_index
    
def least_similar_island(offspring_individual, islands):    
    avg_similarities = [
        np.mean([cosine_similarity(offspring_individual[:n_vars], individual[:n_vars]) for individual in island])
        for island in islands
    ]
    probs = softmax(avg_similarities)
    probs = [1 - p for p in probs]
    probs /= sum(probs)
    return np.random.choice(range(len(probs)), p=probs)

def offspring_gen_placement(num_gen_islands, offspringepoch_gen):
    return [random.randint(0, (num_gen_islands - 1)) for gen_offspring in range(len(offspringepoch_gen))]

# calculates the percintile of the of the offspring's fitness in the population 
def fitness_percentile(fitness, fit_pop):
    offspring_fitness_percentile = percentileofscore(fit_pop, fitness)

    return offspring_fitness_percentile

# weighted average crossover function
def crossover(pop, fit_pop, parents, environment):
    #print("Crossover with dynamic mutation")

    # initialize output array 
    total_offspring = np.zeros((0,n_vars + 1))
    
    for p in range(0,len(parents), 2):

        # select parents
        p1 = pop[parents[p]]
        p2 = pop[parents[p+1]]
        
        n_offspring = 1
        offspring =  np.zeros((n_offspring, n_vars + num_step_size))

        for f in range(1):
            # get weights for weighted average by using softmax
            w1, w2 = softmax([fit_pop[parents[p]], fit_pop[parents[p+1]]])
            # cross_prob = np.random.uniform(0,1)
            offspring[f] = w1 * p1 + p2 * w2

            #check if initializing the stepsize randomly affects the performance of the model

            # calculate the fitness of the offspring and compute it's percentile in the population
            offspring_fitness = simulation(environment, offspring[f])
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
            for i in range(0,len(offspring[f]) - num_step_size):
                if np.random.uniform(0 ,1)<=mutation:
                    offspring[f][i] =   offspring[f][i]+np.random.normal(0, new_step_size)

            offspring[f] = np.array(list(map(lambda y: limits(y), offspring[f])))

            total_offspring = np.vstack((total_offspring, offspring[f]))

    return total_offspring

# weighted average crossover function with static mutation
def crossover_static_mutation(pop, fit_pop, parents):
    #print("Crossover with static mutation")

    # initialize output array 
    total_offspring = np.zeros((0,n_vars + 1))
    
    for p in range(0,len(parents), 2):

        # select parents
        p1 = pop[parents[p]]
        p2 = pop[parents[p+1]]
        
        n_offspring = 1
        offspring =  np.zeros((n_offspring, n_vars + num_step_size))

        for f in range(1):
            # get weights for weighted average by using softmax
            w1, w2 = softmax([fit_pop[parents[p]], fit_pop[parents[p+1]]])
            # cross_prob = np.random.uniform(0,1)
            offspring[f] = w1 * p1 + p2 * w2


            # mutation using updated step size
            for i in range(0,len(offspring[f]) - num_step_size):
                if np.random.uniform(0 ,1)<=mutation:
                    offspring[f][i] =   offspring[f][i]+np.random.normal(0, init_step_size)

            offspring[f] = np.array(list(map(lambda y: limits(y), offspring[f])))

            total_offspring = np.vstack((total_offspring, offspring[f]))

    return total_offspring


def self_adaptive_fitness_based_step_size_mutation(offspring, fit_poplist, environment):
        # calculate the fitness of the offspring and compute it's percentile in the population
    offspring_fitness = simulation(environment, offspring)
    percentile = fitness_percentile(offspring_fitness, fit_poplist[0])

    # obtain the current step_size from the offspring 
    step_size = offspring[n_vars]
    
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
    offspring[n_vars] = new_step_size

    # mutation using updated step size
    for i in range(0,len(offspring) - num_step_size):
        if np.random.uniform(0 ,1)<=mutation:
            offspring[i] =   offspring[i]+np.random.normal(0, new_step_size)

    offspring = np.array(list(map(lambda y: limits(y), offspring)))
    
    return offspring

# weighted average crossover function
def crossover_epoch(poplist, fit_poplist, parentslist, environment, num_gen_islands, generalist_ratio):
    print("Epoch Crossover with dynamic mutation")
    
    # initialize output array for the offspring with certain ratio of the generalist parent.
    total_offspring_gen = np.zeros((0,n_vars + 1))

    # initialize output array for the offspring with random ratios
    total_offspring_all = np.zeros((0,n_vars + 1))
    
    for p in range(0,len(parentslist[0])):

        n_offspring = 1
        offspring_gen =  np.zeros((n_offspring, n_vars + num_step_size))

        # create offspring consisting of a certain guaranteed ratio of the generalist parent
        for f in range(n_offspring):
            ratios = [np.random.uniform(0, 1) for i in range(len(fit_poplist) - num_gen_islands)]
            print("ratios: ")
            print(ratios)
            weights = list(softmax(ratios))
            spec_ratio = 1 - generalist_ratio
            weights = [r * spec_ratio for r in weights]
            # select the generalist parent that will be used for the offspring
            gen_parent = random.randint(0, num_gen_islands - 1)
            # set the weight of the chosen parent to the set ratio while the other generalist parents get weight 0
            for i in range(num_gen_islands):
                if gen_parent == i:
                    weights.insert(i, generalist_ratio)
                else:
                    weights.insert(i, 0)
            for i, w in enumerate(weights):
                print("weight island " + str(i) + ": " + str(w))
            # cross_prob = np.random.uniform(0,1)

            for i, w in enumerate(weights):
                offspring_gen[f] += w * poplist[i][parentslist[i][p]]

            offspring_gen[f] = self_adaptive_fitness_based_step_size_mutation(offspring_gen[f], fit_poplist, environment)

            total_offspring_gen = np.vstack((total_offspring_gen, offspring_gen[f]))


        offspring_all =  np.zeros((n_offspring, n_vars + num_step_size))

        # creating offspring from random ratios of all parents
        for f in range(n_offspring):
            weights = softmax([np.random.uniform(1) for p_idx in range(len(fit_poplist))])
            # cross_prob = np.random.uniform(0,1)
            offspring_all[f] = np.zeros((n_offspring, n_vars + num_step_size))

            for i, w in enumerate(weights):
                offspring_all[f] += w * poplist[i][parentslist[i][p]]

            offspring_all[f] = self_adaptive_fitness_based_step_size_mutation(offspring_all[f], fit_poplist, environment)
            total_offspring_all = np.vstack((total_offspring_all, offspring_all[f]))
    
    print("amount of offspring in the gen: " + str(total_offspring_gen.shape[0]))
    print("amount of offspring in the all: " + str(total_offspring_all.shape[0]))

    return total_offspring_gen, total_offspring_all

def static_mutation(offspring):
    for i in range(0,len(offspring) - num_step_size):
        if np.random.uniform(0 ,1)<=mutation:
            offspring[i] =   offspring[i]+np.random.normal(0, init_step_size)

    offspring = np.array(list(map(lambda y: limits(y), offspring)))
    pass

def crossover_static_mutation_epoch(poplist, fit_poplist, parentslist, num_gen_islands, generalist_ratio):
    print("Epoch Crossover with static mutation")
    
    # initialize output array for the offspring with certain ratio of the generalist parent.
    total_offspring_gen = np.zeros((0,n_vars + 1))

    # initialize output array for the offspring with random ratios
    total_offspring_all = np.zeros((0,n_vars + 1))
    
    for p in range(0,len(parentslist[0])):

        n_offspring = 1
        offspring_gen =  np.zeros((n_offspring, n_vars + num_step_size))

        # create offspring consisting of a certain guaranteed ratio of the generalist parent
        for f in range(n_offspring):            
            ratios = [np.random.uniform(0, 1) for i in range(len(fit_poplist) - num_gen_islands)]
            print("ratios: ")
            print(ratios)
            weights = list(softmax(ratios))
            spec_ratio = 1 - generalist_ratio
            weights = [r * spec_ratio for r in weights]
            # select the generalist parent that will be used for the offspring
            gen_parent = random.randint(0, num_gen_islands - 1)
            # set the weight of the chosen parent to the set ratio while the other generalist parents get weight 0
            for i in range(num_gen_islands):
                if gen_parent == i:
                    weights.insert(i, generalist_ratio)
                else:
                    weights.insert(i, 0)
            for i, w in enumerate(weights):
                print("weight island " + str(i) + ": " + str(w))
            # cross_prob = np.random.uniform(0,1)

            for i, w in enumerate(weights):
                offspring_gen[f] += w * poplist[i][parentslist[i][p]]

            offspring_gen[f] = static_mutation(offspring_gen[f])

            total_offspring_gen = np.vstack((total_offspring_gen, offspring_gen[f]))


        offspring_all =  np.zeros((n_offspring, n_vars + num_step_size))

        # creating offspring from random ratios of all parents
        for f in range(n_offspring):
            weights = softmax([np.random.uniform(1) for p_idx in range(len(fit_poplist))])
            # cross_prob = np.random.uniform(0,1)
            offspring_all[f] = np.zeros((n_offspring, n_vars + num_step_size))

            for i, w in enumerate(weights):
                offspring_all[f] += w * poplist[i][parentslist[i][p]]

            offspring_all[f] = static_mutation(offspring_all[f])
            total_offspring_all = np.vstack((total_offspring_all, offspring_all[f]))
    
    print("amount of offspring in the gen: " + str(total_offspring_gen.shape[0]))
    print("amount of offspring in the all: " + str(total_offspring_all.shape[0]))

    return total_offspring_gen, total_offspring_all

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
    slicesize = int(slicesize)
    nparents = int(nparents)
    combined = list(zip(fit,indices))

    sorted_combined = sorted(combined, key = lambda x:x[0], reverse=True) #highest first
    sorted_population = [ind for _, ind in sorted_combined]
    best_individuals = sorted_population[:slicesize]
    other_ind = sorted_population[slicesize:]
    random_ind = list(np.random.choice(other_ind, nparents - slicesize, False))

    return best_individuals + random_ind

# evolution
def evolve_generation(pop, fit_pop, env):
    parents = truncation_random_hybrid_selection(pop, fit_pop, pop.shape[0] // 2, ratio)
    random.shuffle(parents)
    offspring = crossover(pop, fit_pop, parents, env) if experiment_type == "dynamic" else crossover_static_mutation(pop, fit_pop, parents) # crossover
    fit_offspring = evaluate(env, offspring)   # evaluation
    pop = np.vstack((pop,offspring))
    fit_pop = np.append(fit_pop,fit_offspring)

    best = np.argmax(fit_pop) #best solution in generation
    fit_pop[best] = float(evaluate(env, np.array([pop[best] ]))[0]) # repeats best eval, for stability issues
    best_sol = fit_pop[best]

    pop, fit_pop = survival_selection(pop, fit_pop, npop)
    solutions = [pop, fit_pop]
    env.update_solutions(solutions)

    return pop, fit_pop, best

def evolve_epoch(pop_gens, fit_pop_gens, pop_specs, fit_pop_specs, env_gens, env_specs, ratio = 0.8):
    nparents = 20 #min([pop.shape[0] // (1/parent_ratio) for pop in pop_specs + pop_gens])
    parents = []
    for i, pop in enumerate(pop_specs):
        pop_parents = truncation_random_hybrid_selection(pop, fit_pop_specs[i], nparents, ratio)
        random.shuffle(pop_parents)
        parents.append(pop_parents)
    
    for i, pop in enumerate(pop_gens):
        pop_parents = truncation_random_hybrid_selection(pop, fit_pop_gens[i], nparents, ratio)
        random.shuffle(pop_parents)
        parents.append(pop_parents)    
    
    combined_pop = pop_gens + pop_specs
    combined_fit_pop = fit_pop_gens + fit_pop_specs
    combined_env = env_gens + env_specs
    num_gen_islands = len(env_gens)

    generalist_ratio = start_val_gen_ratio + ((end_val_gen_ratio - start_val_gen_ratio) / total_epochs) * epoch_number
    print("generalist ratio: " + str(generalist_ratio))

    offspringepoch_gen, offspringepoch_all = crossover_epoch(combined_pop, combined_fit_pop, 
                                    parents, env_gens[0], num_gen_islands, generalist_ratio) if experiment_type == "dynamic" else crossover_static_mutation_epoch(
                                    combined_pop, combined_fit_pop, parents, num_gen_islands, generalist_ratio)
    #offspring_placements = [similar_island(individual, combined_pop, 0.8) for individual in offspringepoch]
    offspring_all_placements = [similar_island(individual, combined_pop) if similar_migration else least_similar_island(individual, combined_pop) for individual in offspringepoch_all]
    offspring_gen_placements = offspring_gen_placement(num_gen_islands, offspringepoch_gen)

    total_offspring = np.vstack((offspringepoch_gen, offspringepoch_all))
    offspring_placements = offspring_gen_placements + offspring_all_placements

    print("offspring placements: " + str(offspring_placements))

    for i, (pop_total, fit_pop_total, env) in enumerate(zip(combined_pop, combined_fit_pop, combined_env)):
        pop_similars = []
        for idx, individual in enumerate(total_offspring):
            if offspring_placements[idx] == i:
                pop_similars.append(individual)
        if len(pop_similars) != 0:
            fit_offspringepoch = evaluate(env, pop_similars)
            pop_total = np.vstack((pop_total, pop_similars))
            fit_pop_total = np.append(fit_pop_total, fit_offspringepoch)
            pop_total, fit_pop_total = survival_selection(pop_total, fit_pop_total, npop, True)
            solutions = [pop_total, fit_pop_total]
            env.update_solutions(solutions)
        
    for env_gen in env_gens:
        env_gen.save_state()

def pop_std(pop):
    return np.mean(np.std(np.stack(pop), axis=0))

# loads file with the best solution for testing
if run_mode =='test':

    bsol = np.loadtxt(experiment_name+'/best.txt')
    print( '\n RUNNING SAVED BEST SOLUTION \n')
    env_general.update_parameter('speed','normal')
    env_general.update_parameter('visuals',True)
    x = evaluate(env_general,[bsol])
    print(x)

    sys.exit(0)

for run in range(10):

# initializes population loading old solutions or generating new ones
    best = 0
    mean = 0
    std = 0


    print( '\nNEW EVOLUTION\n')

    env_gens, env_specs = init_islands()
    env_general = env_gens[0] # Hardcoded for now.
    pop_gens = []
    pop_specs = []
    fit_pop_gens = []
    fit_pop_specs = []
    best_gen = []
    mean_gen = []
    std_gen = []
    ini_g = 0

    # loads file with the best solution for testing
    if run_mode =='test':

        bsol = np.loadtxt(experiment_name+'/best.txt')
        print( '\n RUNNING SAVED BEST SOLUTION \n')
        env_general.update_parameter('speed','normal')
        evaluate([bsol])

        sys.exit(0)

    for i, env_gen in enumerate(env_gens):
        print(f'Initializing the {i + 1}. generalist population\n')
        pop = np.random.uniform(dom_l, dom_u, (npop, n_vars))
        pop = np.hstack([pop, np.full((pop.shape[0], 1), init_step_size)])
        fit_pop = evaluate(env_gen, pop)
        best_gen.append(np.argmax(fit_pop))
        mean_gen.append(np.mean(fit_pop))
        std_gen.append(np.std(fit_pop))
        pop_gens.append(pop)
        fit_pop_gens.append(fit_pop)
        solutions = [pop, fit_pop]
        env_gen.update_solutions(solutions)

    for i, env_spec in enumerate(env_specs):
        print(f'Initializing the {i + 1}. specialist population\n')
        pop = np.random.uniform(dom_l, dom_u, (npop, n_vars))
        pop = np.hstack([pop, np.full((pop.shape[0], 1), init_step_size)])
        fit_pop = evaluate(env_spec, pop)
        pop_specs.append(pop)
        fit_pop_specs.append(fit_pop)
        solutions = [pop, fit_pop]
        env_spec.update_solutions(solutions)

    best_gen_idx = np.argmax([fit_pop_gen[best_gen[i]] for i, fit_pop_gen in enumerate(fit_pop_gens)])
    best = best_gen[best_gen_idx]
    mean = mean_gen[best_gen_idx]
    std = std_gen[best_gen_idx]

    # saves results for first pop
    file_aux  = open(experiment_name+f'/results{run}.txt','a')
    file_aux.write('\n\ngen best mean std div')


    print( '\n GENERATION '+str(ini_g)+' '+str(round(fit_pop_gens[best_gen_idx][best],6))+' '+str(round(mean,6))+' '+str(round(std,6))+' '+str(round(pop_std(pop_gens[0]),6)))
    file_aux.write('\n'+str(ini_g)+' '+str(round(fit_pop_gens[best_gen_idx][best],6))+' '+str(round(mean,6))+' '+str(round(std,6)) +' '+str(round(pop_std(pop_gens[0]),6))  )
    file_aux.close()       



    for i in range(ini_g+1, gens):

        for j, (pop, fit_pop, env) in enumerate(zip(pop_gens, fit_pop_gens, env_gens)):
            pop, fit_pop, best = evolve_generation(pop, fit_pop, env)
            best_gen[j] = np.argmax(fit_pop)
            mean_gen[j] = np.mean(fit_pop)
            std_gen[j] = np.std(fit_pop)
            pop_gens[j] = pop
            fit_pop_gens[j] = fit_pop

        for k, (pop, fit_pop, env) in enumerate(zip(pop_specs, fit_pop_specs, env_specs)):
            pop, fit_pop, _ = evolve_generation(pop, fit_pop, env)
            pop_specs[k] = pop
            fit_pop_specs[k] = fit_pop

        best_gen_idx = np.argmax([fit_pop_gen[best_gen[k]] for k, fit_pop_gen in enumerate(fit_pop_gens)])
        best = best_gen[best_gen_idx]
        mean = mean_gen[best_gen_idx]
        std = std_gen[best_gen_idx]

        if i == gens - 1:
            for l, (pop, fit_pop) in enumerate(zip(pop_specs, fit_pop_specs)):
                fit_pop = evaluate(env_gens[0], pop)
                best_ind = np.argmax(fit_pop)
                pop_gens.append(pop)
                fit_pop_gens.append(fit_pop)
                best_gen.append(best_ind)
            best_gen_idx = np.argmax([fit_pop_gen[best_gen[k]] for k, fit_pop_gen in enumerate(fit_pop_gens)])
            best = best_gen[best_gen_idx]

        # saves results
        file_aux  = open(experiment_name+f'/results{run}.txt','a')
        print( '\n GENERATION '+str(i)+' '+str(round(fit_pop_gens[best_gen_idx][best],6))+' '+str(round(mean,6))+' '+str(round(std,6))+' '+str(round(pop_std(pop_gens[0]),6)))
        file_aux.write('\n'+str(i)+' '+str(round(fit_pop_gens[best_gen_idx][best],6))+' '+str(round(mean,6))+' '+str(round(std,6)) +' '+str(round(pop_std(pop_gens[0]),6))  )
        file_aux.close()

        # saves generation number
        file_aux  = open(experiment_name+f'/gen{run}.txt','w')
        file_aux.write(str(i))
        file_aux.close()

        # saves file with the best solution
        np.savetxt(experiment_name+f'/best{run}.txt',pop_gens[best_gen_idx][best])

        if i % epoch_frequency == 0:
            epoch_number += 1
            evolve_epoch(pop_gens, fit_pop_gens, pop_specs, fit_pop_specs, env_gens, env_specs)




    fim = time.time() # prints total execution time for experiment
    print( '\nExecution time: '+str(round((fim-ini)/60))+' minutes \n')
    print( '\nExecution time: '+str(round((fim-ini)))+' seconds \n')




    env_gens[0].state_to_log() # checks environment state

        