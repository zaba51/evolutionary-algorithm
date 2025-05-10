import numpy as np
import copy


def invert(obj: np.ndarray):
    i, j = sorted(np.random.choice(len(obj), size=2, replace=False))
    return np.concatenate((obj[:i], obj[i:j][::-1], obj[j:]))


def invert_pop(pop: np.ndarray, p_invert=0.3):
    if p_invert <= 0:
        return pop

    new_pop = copy.deepcopy(pop)

    for i in range(len(new_pop)):
        if np.random.rand() <= p_invert:
            new_pop[i] = invert(new_pop[i])

    return new_pop
