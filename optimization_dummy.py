###############################################################################
# EvoMan FrameWork - V1.0 2016  			                                  #
# DEMO : Neuroevolution - Genetic Algorithm  neural network.                  #
# Author: Karine Miras        			                                      #
# karine.smiras@gmail.com     				                                  #
###############################################################################

# imports framework
import sys

from evoman.environment import Environment
from demo_controller import player_controller

# imports other libs
import numpy as np
import os

# runs simulation
def simulation(env,x):
    f,p,e,t = env.play(pcont=x)
    return f

# evaluation
def evaluate(env, x):
    return np.array(list(map(lambda y: simulation(env,y), x)))


def main():
    # choose this for not using visuals and thus making experiments faster
    headless = True
    if headless:
        os.environ["SDL_VIDEODRIVER"] = "dummy"


    experiment_name = 'optimization_test'
    if not os.path.exists(experiment_name):
        os.makedirs(experiment_name)

    n_hidden_neurons = 10

    # initializes simulation in individual evolution mode, for single static enemy.
    env = Environment(experiment_name=experiment_name,
                    enemies=[2],
                    playermode="ai",
                    player_controller=player_controller(n_hidden_neurons), # you  can insert your own controller here
                    enemymode="static",
                    level=2,
                    speed="fastest",
                    visuals=False)


    # number of weights for multilayer with 10 hidden neurons
    n_vars = (env.get_num_sensors()+1)*n_hidden_neurons + (n_hidden_neurons+1)*5

    # start writing your own code from here
    def survival_selection(pop, fit_pop, mut_pop, num_selected):
        min_fitness = np.min(fit_pop)
        # shifting the fitness values to make sure it contains no negative
        if min_fitness < 0:
            fit_pop = fit_pop - min_fitness
        
        probs = fit_pop / np.sum(fit_pop)

        best_idx = np.argmax(fit_pop)
        best_nn = pop[best_idx]
        best_fit = fit_pop[best_idx]
        best_mut = mut_pop[best_idx]

        num_selected = num_selected - 1
        
        if num_selected > (pop.shape[0] - 1):
            num_selected = (pop.shape[0] - 1)
        
        indices = np.random.choice(pop.shape[0], num_selected, p = probs, replace=False)
        selected_pop = pop[indices]
        selected_fit = fit_pop[indices]
        selected_mut = mut_pop[indices]

        selected_pop = np.vstack([selected_pop, best_nn])
        selected_fit = np.vstack([selected_fit, best_fit])
        selected_mut = np.vstack([selected_mut, best_mut])
        
        return selected_pop, selected_fit, selected_mut



if __name__ == '__main__':
    main()