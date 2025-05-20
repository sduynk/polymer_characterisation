import pandas as pd
import numpy as np
from scipy.optimize import minimize
import torch
import torch.nn as nn 
import random
import os 

def RA(point, mean):

    d1, p1, h1 = point
    d2, p2, h2 = mean

    return np.sqrt(4*(d1 - d2)**2 + (p1 - p2)**2 + (h1 - h2)**2)

def vebber_cost(center, radius, positive_points, negative_points):
    d, p, h = center

    if d < 10 or d > 20:
        return -1e10
    
    if p < 0 or p > 25.5:
        return -1e10
    
    if h < 0 or h > 42.5:
        return -1e10
    
    N = len(positive_points) + len(negative_points)

    all_points = np.concatenate((positive_points, negative_points))
    all_labels = np.concatenate((np.ones(len(positive_points)), 0*np.ones(len(negative_points))))
    all_ras = [RA(point, center) for point in all_points]
    all_costs = np.zeros(len(all_ras))

    for i in range(len(all_ras)):
        if all_ras[i] < radius:
            if all_labels[i] == 1:
                all_costs[i] = 1
            else:
                all_costs[i] = np.e ** -(radius - all_ras[i])
        else:
            if all_labels[i] == 1:
                all_costs[i] = np.e ** -(all_ras[i] - radius)
            else:
                all_costs[i] = 1
    
    all_cost = np.prod(all_costs, axis=0) ** (1/N)
    all_cost = all_cost * (radius ** (-1/20))
    return all_cost


def init_pop(population_size, positive_points):
    

    # Calculate the mean of the positive points
    mean_positive = np.mean(positive_points, axis=0)

    # Calculate the radius as the distance to the furthest point from the mean
    distances = np.linalg.norm(positive_points - mean_positive, axis=1)
    radius = np.max(distances)

    # Generate the initial population
    population = np.random.normal(loc=np.append(mean_positive, radius), scale=1.0, size=(population_size, 4))

    population = np.clip(population, a_min=0, a_max=20)

    return population

def genetic_algorithm(population_size, generations, mutation_rate, positive_points, negative_points, loss_function, mode="maximize"):
    """
    Genetic Algorithm implementation with support for maximization or minimization.
    
    Parameters:
    - population_size: Number of individuals in the population.
    - generations: Number of generations to run the algorithm.
    - mutation_rate: Probability of mutation.
    - positive_points: Positive data points.
    - negative_points: Negative data points.
    - loss_function: The objective function to optimize.
    - mode: "maximize" or "minimize" (default is "maximize").
    """

    # Set random seeds for reproducibility
    np.random.seed(0)
    random.seed(0)

    # Set initial best fitness based on the mode
    if mode == "maximize":
        best_fitness = float('-inf')
    elif mode == "minimize":
        best_fitness = float('inf')
    else:
        raise ValueError("Mode must be 'maximize' or 'minimize'")
    
    best_individual = None

    # Generate random initial population
    population = init_pop(population_size, positive_points)

    for generation in range(generations):
        # Compute fitness for each individual
        fitness_scores = np.array([loss_function(ind[:3], ind[3], positive_points, negative_points) for ind in population])

        # Adjust fitness scores for minimization if needed
        if mode == "minimize":
            fitness_scores = -fitness_scores

        # Select the top 1024 individuals as parents
        top_indices = np.argsort(fitness_scores)[-(1024):]
        parents = population[top_indices]

        # Update best individual if found a better one
        if mode == "maximize" and fitness_scores[top_indices[-1]] > best_fitness:
            best_fitness = fitness_scores[top_indices[-1]]
            best_individual = parents[-1]
        elif mode == "minimize" and -fitness_scores[top_indices[-1]] < best_fitness:
            best_fitness = -fitness_scores[top_indices[-1]]
            best_individual = parents[-1]

        # Mutate 95% of parents
        num_to_mutate = int(0.95 * len(parents))
        for i in range(num_to_mutate):
            if np.random.rand() < mutation_rate:
                mutation_indices = np.random.choice(4, size=4, replace=False)  # Mutate all 4 genes
                parents[i][mutation_indices] += np.random.uniform(-1, 1, size=4)  # Random top-limited mutation

        # Generate children through crossover
        children = []
        for _ in range(population_size):
            # Select two random parents
            parent1, parent2 = parents[np.random.choice(len(parents), 2, replace=False)]

            # Perform two-point crossover
            crossover_points = sorted(np.random.choice(range(1, 4), size=2, replace=False))
            child = np.concatenate((
                parent1[:crossover_points[0]],
                parent2[crossover_points[0]:crossover_points[1]],
                parent1[crossover_points[1]:]
            ))

            children.append(child)


        # Convert children list to numpy array
        children = np.array(children)
        # Clip values to ensure they are positive
        children = np.clip(children, a_min=0, a_max=None)

        # Combine parents and children to form the new population
        population = children

        if generation % 100 == 0:
            print(f"Generation {generation}: Best Fitness = {best_fitness}")
            print(f"Best Individual = {best_individual}")

    # Return the best individual and its fitness (adjusted for minimization if needed)
    if mode == "minimize":
        return best_individual, -best_fitness
    return best_individual, best_fitness


csvs = ["PVP.csv", "PMMA.csv", "PCL.csv", "PS.csv"]

hsp = {
    "PVP": [17.5, 8.0, 15.0],
    "PMMA": [18.6, 10.5, 5.1],
    "PCL": [17.7, 5.0, 8.4],
    "PS": [18.5, 4.5, 2.9],
    "PES": [19.0, 11.0, 8.0]
}

population_size = 8192
generations = 200
mutation_rate = 0.95

loss_functions = [vebber_cost]

rows = []

def distance(point1, point2):
    return np.sqrt(np.sum((point1 - point2) ** 2))

def percentage_distance_error(distance, target):
    norm = np.linalg.norm(target)
    return (distance / norm) * 100

for csv in csvs:

    for loss in loss_functions:

        print(csv)

        df = pd.read_csv(csv)

        # Example data
        class_in = df[df['Solub'] == 1][['delta_d', 'delta_p', 'delta_h']].values
        class_out = df[df['Solub'] == 0][['delta_d', 'delta_p', 'delta_h']].values

        positive_points = class_in
        negative_points = class_out

        mode = "maximize"

        # Run the genetic algorithm
        best_individual, best_fitness = genetic_algorithm(population_size, generations, mutation_rate, positive_points, negative_points, loss, mode=mode)

        d, p, h, R0 = best_individual

        target = hsp[csv.split(".")[0].split("_")[0]]
        D, P, H = target

        target = np.array(target)
        pred = np.array([d, p, h])
        
        error = distance(target, pred)
        pe = percentage_distance_error(error, target)

        row = {"loss": loss.__name__, "dataset": csv.split(".")[0], "d": d, "p": p, "h": h, "R0": R0, "D": D, "P": P, "H":H, "fitness": best_fitness, "distance": error, "percentage_error": pe}

        rows.append(row)

df = pd.DataFrame(rows)





df


