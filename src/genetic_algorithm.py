from cross_methods import cross
from inversion import invert_pop
from mutation import mutate
from population import generate_population, evaluate_population, get_elites, get_best, select
import numpy as np
import random


class GeneticAlgorithm:
    def __init__(self, objective_function, population_size, n_generations, bounds, N=2,
           precision=6, selection_method='best', selection_ratio=0.3, tournament_size=3, 
           mutation_method = 'gauss', p_mutation=0.7,
           n_mutation_points=1, sigma=0.1, n_elites=1, cross_method = 'average', 
           alpha=0.25, beta=0.5, p_inversion=0.0):
        self.objective_function = objective_function
        self.population_size = population_size
        self.n_generations = n_generations
        self.bounds = bounds
        self.N = N
        self.precision = precision
        self.selection_method = selection_method
        self.selection_ratio = selection_ratio
        self.tournament_size = tournament_size
        self.mutation_method = mutation_method
        self.p_mutation = p_mutation
        self.n_mutation_points = n_mutation_points
        self.n_elites = n_elites
        self.cross_method = cross_method
        self.sigma = sigma
        self.alpha = alpha
        self.beta = beta
        self.p_inversion = p_inversion

        self.population_history = []
        self.get_history_step = n_generations // 5
        self.best_values_history = []

    def evolve(self):
        best_generation = 0

        population = generate_population(self.population_size, self.N, self.bounds)
        evaluated_population = evaluate_population(self.objective_function, population)
        elites, best_indices = get_elites(population, evaluated_population, self.n_elites, False)
        best_solution, best_value = get_best(population, evaluated_population, False)

        self.population_history.append((0, population.copy(), evaluated_population.copy()))
        self.best_values_history.append(best_value)

        for i in range(self.n_generations):
            new_population = np.empty((0, population.shape[1]), dtype=int)

            # Select % of the previous population
            selected = select(population,
                              evaluated_population,
                              self.selection_method,
                              self.selection_ratio, self
                              .tournament_size,
                              False)

            editable_population_size = self.population_size - self.n_elites

            # Cross
            while len(new_population) < editable_population_size:
                random_parents = random.sample(list(selected), 2)
                children = cross(*random_parents, method=self.cross_method, bounds = self.bounds, alpha=self.alpha, beta=self.beta, obj_func=self.objective_function)
                children_to_add = children[:(editable_population_size - len(new_population))]
                new_population = np.vstack((new_population, np.array(children_to_add)))

            assert (len(new_population) == editable_population_size), \
                f"Expected population size: {editable_population_size}. Found: {len(new_population)}."

            # Mutate
            new_population = mutate(new_population, self.p_mutation, self.mutation_method, self.n_mutation_points, self.bounds, self.sigma)

            # Invert
            new_population = invert_pop(new_population, self.p_inversion)

            # Merge previous elites with new popuation
            new_population = np.vstack((new_population, elites))
            population = new_population

            assert population.shape[0] == self.population_size and population.shape[1] == self.N , \
                f"Population size mismatch. Expected: {(self.population_size, self.N)}. Found: {population.shape}."

            evaluated_population = evaluate_population(self.objective_function,
                                                       population)
            elites, elites_indices = get_elites(population, evaluated_population, self.n_elites, False)
            best_solution_from_generation, best_value_from_generation = get_best(population,
                                                                                 evaluated_population,
                                                                                 False)
            self.best_values_history.append(best_value_from_generation)

            if i % self.get_history_step == 0:
                self.population_history.append((i, population.copy(), evaluated_population.copy()))

            if best_value_from_generation < best_value:
                best_generation = i
                best_solution = best_solution_from_generation
                best_value = best_value_from_generation

            if i == 0 or (i + 1) % 10 == 0:
                print(f"Gen: {i + 1}. Wynik: {best_value}")

        self.population_history.append((self.n_generations, population.copy(), evaluated_population.copy()))

        return { 'best_solution': best_solution, 'best_value': best_value, 'best_generation': best_generation }
