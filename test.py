import numpy as np

population = ["p0","p1","p2","p3","p4","p5","p6","p7","p8","p9","p10","p11","p12","p13","p14","p15","p16","p17","p18","p19"]
popfitness = [17,   3,   19,  5,   24,  8,   9,   11,  19,  6,   35,   10,   4,    14,   12,   8,    21,   7,    15,   6]

def truncation_random_hybrid_selection(pop,fit, nparents, ratio):
    """
    Returns a list of parents to be used in the crossover. The indices to the parents are provided.
    Uses a mix of Truncation Selection and random selection to generate a list of parents.
    Through truncation selection a certain number of best individuals are selected as parents.
    The parent list is then filled up by randomly sampling from the remaining individuals.
    pop: list that contains the population which will be sampled
    fit: list that contains the corresponding fitness values of the population
    nparents: the amount of parents to be generated
    ratio: the ratio between the parents chosen through truncation and random selection.
    For example a ratio of 3 will yield 1/3 selected through truncation and 2/3 selected randomly.
    """
    popsize = len(pop)
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

print(truncation_random_hybrid_selection(population, popfitness, 10, 0.33))