from bisect import bisect_right
from collections import defaultdict
from copy import deepcopy
from functools import partial
from itertools import chain
from operator import eq
def decFunc(func):

    def wrapFunc(*args, **kargs):
        individuals = func(*args, **kargs)
        self.update(individuals)
        return individuals
    return wrapFunc