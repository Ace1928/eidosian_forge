import sys
from math import sqrt, erfc, log, floor
from heapq import heapify, heappop, heappush
from itertools import permutations
from collections import defaultdict, Counter
import numpy as np
from Bio.Align import Alignment
from Bio.Data import CodonTable
def func_w1(params, pi=pi, codon_cnt=codon_cnt, codons=codons, codon_table=codon_table):
    """Temporary function, params = [t, k]. w is fixed to 1."""
    return -_likelihood_func(params[0], params[1], 1.0, pi, codon_cnt, codons=codons, codon_table=codon_table)