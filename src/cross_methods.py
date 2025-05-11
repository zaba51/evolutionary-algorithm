import numpy as np
import math

from src.population import evaluate_population, get_elites


def cross(parent1, parent2, method="arithmetic", alpha=0.25, beta=0.5, bounds = [-math.inf, math.inf], obj_func = None)-> np.ndarray:
    if method == "arithmetic":
        children = arithmetic_cross(parent1, parent2, alpha)
    elif method == "blend_alpha":
        children = blend_cross_alpha(parent1, parent2, alpha)
    elif method == 'linear':
        children = linear_cross(parent1, parent2, obj_func)
    elif method == "blend_alpha_beta":
        children = blend_cross_alpha_beta(parent1, parent2, alpha, beta)
    elif method == "average":
        children = (average_cross(parent1, parent2), )
    else:
        raise ValueError()

    bounded_children = tuple(np.clip(child, bounds[0], bounds[1]) for child in children if child is not None)
    return bounded_children

def arithmetic_cross(parent1: np.ndarray, parent2: np.ndarray, alpha: float = None) -> tuple[np.ndarray, np.ndarray]:
    child1 = alpha * parent1 + (1 - alpha) * parent2
    child2 = (1 - alpha) * parent1 + alpha * parent2
    return child1, child2

def linear_cross(parent1: np.ndarray, parent2: np.ndarray, obj_func) -> tuple[np.ndarray, np.ndarray]:
    assert obj_func is not None, "obj_func can not be none for linear cross"

    child1 = 0.5 * (parent1 + parent2)
    child2 = 1.5 * parent1 - 0.5 * parent2
    child3 = -0.5 * parent1 + 1.5 * parent2

    pop = np.array([child1, child2, child3])
    evaluated = evaluate_population(obj_func, pop)
    elites = get_elites(pop, evaluated, 2, False)

    return elites[0]

def blend_cross_alpha(parent1: np.ndarray, parent2: np.ndarray, alpha: float = None) -> tuple[np.ndarray, np.ndarray]:
    c_min = np.minimum(parent1, parent2)
    c_max = np.maximum(parent1, parent2)
    d = np.abs(c_max - c_min)
    lower_bound = c_min - alpha * d
    upper_bound = c_max + alpha * d

    child1 = np.random.uniform(lower_bound, upper_bound)
    child2 = np.random.uniform(lower_bound, upper_bound)
    return child1, child2

def blend_cross_alpha_beta(parent1: np.ndarray, parent2: np.ndarray, alpha: float = None, beta: float = None) -> tuple[np.ndarray, np.ndarray]:
    c_min = np.minimum(parent1, parent2)
    c_max = np.maximum(parent1, parent2)
    d = np.abs(c_max - c_min)
    lower_bound = c_min - alpha * d
    upper_bound = c_max + beta * d

    child1 = np.random.uniform(lower_bound, upper_bound)
    child2 = np.random.uniform(lower_bound, upper_bound)
    return child1, child2

def average_cross(parent1: np.ndarray, parent2: np.ndarray) -> tuple[np.ndarray]:
    child = (parent1 + parent2) / 2
    return child