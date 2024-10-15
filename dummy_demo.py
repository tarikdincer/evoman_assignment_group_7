################################
# EvoMan FrameWork - V1.0 2016 #
# Author: Karine Miras         #
# karine.smiras@gmail.com      #
################################

# imports framework
import sys, os

from evoman.environment import Environment
from demo_controller import player_controller

# imports other libs
import numpy as np

experiment_name = 'dummy_demo'
if not os.path.exists(experiment_name):
    os.makedirs(experiment_name)

# Update the number of neurons for this specific example
n_hidden_neurons = 10

# initializes environment for single objective mode (specialist)  with static enemy and ai player
env = Environment(experiment_name=experiment_name,
				  playermode="ai",
                  multiplemode="yes",
				  player_controller=player_controller(n_hidden_neurons),
			  	  speed="normal",
				  enemymode="static",
				  level=2,
				  visuals=True)


# tests saved demo solutions for each enemy

#Update the enemy
env.update_parameter('enemies',[1,2,3,4,5,6,7,8])

	# Load specialist controller
sol = np.loadtxt('optimization_test_task/best.txt')
print('\n LOADING SAVED SPECIALIST SOLUTION FOR ALL ENEMIES \n')
env.play(sol)