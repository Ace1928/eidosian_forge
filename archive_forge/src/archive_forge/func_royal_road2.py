from functools import wraps
def royal_road2(individual, order):
    """Royal Road Function R2 as presented by Melanie Mitchell in :
    "An introduction to Genetic Algorithms".
    """
    total = 0
    norder = order
    while norder < order ** 2:
        total += royal_road1(individual, norder)[0]
        norder *= 2
    return (total,)