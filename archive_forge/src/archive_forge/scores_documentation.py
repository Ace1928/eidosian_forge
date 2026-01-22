import operator
from functools import reduce
from math import fabs
from random import shuffle
from nltk.util import LazyConcatenation, LazyMap

    Returns an approximate significance level between two lists of
    independently generated test values.

    Approximate randomization calculates significance by randomly drawing
    from a sample of the possible permutations. At the limit of the number
    of possible permutations, the significance level is exact. The
    approximate significance level is the sample mean number of times the
    statistic of the permutated lists varies from the actual statistic of
    the unpermuted argument lists.

    :return: a tuple containing an approximate significance level, the count
             of the number of times the pseudo-statistic varied from the
             actual statistic, and the number of shuffles
    :rtype: tuple
    :param a: a list of test values
    :type a: list
    :param b: another list of independently generated test values
    :type b: list
    