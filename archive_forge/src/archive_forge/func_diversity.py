import math
import itertools
import random
def diversity(population):
    nind = len(population)
    ndim = len(population[0])
    d = [0.0] * ndim
    for x in population:
        d = [di + xi for di, xi in zip(d, x)]
    d = [di / nind for di in d]
    return math.sqrt(sum(((di - xi) ** 2 for x in population for di, xi in zip(d, x))))