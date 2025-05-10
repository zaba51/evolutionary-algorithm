import numpy as np
import copy
import math

def mutate_uniform(obj: np.ndarray, n: int = 1, bounds = [-math.inf, math.inf] ):
    mutate_indices = np.random.choice(len(obj), size=n, replace=False)
    obj[mutate_indices] = np.random.uniform(low=bounds[0], high=bounds[1], size=n)

def mutate_gauss(obj: np.ndarray, n: int = 1, bounds = [-np.inf, np.inf], sigma: float = 0.1):
    mutate_indices = np.random.choice(len(obj), size=n, replace=False)
    gauss = np.random.normal(loc=0.0, scale=sigma, size=n)
    obj[mutate_indices] += gauss
    obj[mutate_indices] = np.clip(obj[mutate_indices], bounds[0], bounds[1])

def mutate(pop: np.ndarray, pm: float, method = 'gauss', n: int = 1, bounds = [-math.inf, math.inf], sigma=0.1 ):
  if (pm <= 0):
    return pop

  new_pop = copy.deepcopy(pop)
  for obj in new_pop:
      if np.random.rand() <= pm:
          if method == 'gauss':
            mutate_gauss(obj, n, bounds, sigma)
          if method == 'uniform':
            mutate_uniform(obj, n, bounds)

  return new_pop