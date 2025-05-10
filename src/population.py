import math
import random
import numpy as np

def generate_population(pop_size: int, n_variables: int, bounds: tuple[float, float]) -> np.ndarray:
    low, high = bounds
    pop = np.random.uniform(low, high, size=(pop_size, n_variables))
    return pop

def evaluate_population(func, pop):
    evaluated_pop = np.apply_along_axis(func, 1, pop)

    return evaluated_pop

def get_best(pop, evaluated_pop, max=True):
    best_index = np.argmax(evaluated_pop) if max else np.argmin(evaluated_pop)

    best_individual = pop[best_index]
    best_value = evaluated_pop[best_index]
    return best_individual, best_value


def get_elites(pop: np.ndarray, evaluated_pop: np.ndarray, n: int, maximize: bool) -> tuple:
    if n <= 0:
        return np.empty((0, pop.shape[1]), dtype=int), np.empty((0, 1), dtype=int)

    n = min(n, len(pop))
    indices_asc = np.argsort(evaluated_pop)
    best_indices = indices_asc[::-1][:n] if maximize else indices_asc[:n]

    elites = pop[best_indices]
    return elites, best_indices


def select(pop, pop_results, method='best', selection_ratio=0.3, tournament_size=3, max=True):
    n = math.ceil(selection_ratio * len(pop_results))

    if method == "best":
        return best(pop, pop_results, n, max)
    elif method == "roulette":
        return roulette(pop, pop_results, n, max)
    elif method == "tournament":
        return tournament(pop, pop_results, n, tournament_size, max)
    else:
        raise ValueError()


def best(pop, pop_results, n, max=True):
    return get_elites(pop, pop_results, n, False)[0]


def roulette(pop, pop_results, n_selected, max=True):
    evaluated_pop = pop_results if max else pop_results ** (-1)

    sum_all = np.sum(evaluated_pop)
    probabilities = evaluated_pop / sum_all

    assert np.isclose(np.sum(probabilities), 1), f"Sum of propabilities does not equal 1. Found: {sum(probabilities)}"

    wheel = np.zeros_like(evaluated_pop)
    wheel[0] = probabilities[0]
    for i in range(1, len(probabilities)):
        wheel[i] = wheel[i - 1] + probabilities[i]

    selected = np.zeros_like(pop)[:n_selected]
    for i in range(n_selected):
        rand_num = random.random()
        for j in range(len(wheel)):
            if rand_num <= wheel[j]:
                selected[i] = pop[j]
                break

    return selected


def tournament(pop, pop_results, num_selected, tournament_size, max=True):
    selected = []
    available_indices = list(range(len(pop)))

    for _ in range(num_selected):
        if not available_indices:
            break

        tournament_indices = random.sample(available_indices, min(tournament_size, len(available_indices)))
        tournament_contestants = pop[tournament_indices]
        tournament_scores = pop_results[tournament_indices]

        best_idx = np.argmax(tournament_scores) if max else np.argmin(tournament_scores)
        selected.append(tournament_contestants[best_idx])

        available_indices.remove(tournament_indices[best_idx])

    return np.array(selected)
