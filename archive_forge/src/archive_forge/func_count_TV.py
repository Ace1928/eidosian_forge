import sys
from math import sqrt, erfc, log, floor
from heapq import heapify, heappop, heappush
from itertools import permutations
from collections import defaultdict, Counter
import numpy as np
from Bio.Align import Alignment
from Bio.Data import CodonTable
def count_TV(codon1, codon2, diff, codon_table, weight=1):
    purine = ('A', 'G')
    pyrimidine = ('T', 'C')
    dic = codon_table.forward_table
    stop = codon_table.stop_codons
    if codon1 in stop or codon2 in stop:
        if codon1[diff] in purine and codon2[diff] in purine:
            return [0, 0, weight, 0]
        elif codon1[diff] in pyrimidine and codon2[diff] in pyrimidine:
            return [0, 0, weight, 0]
        else:
            return [0, 0, 0, weight]
    elif dic[codon1] == dic[codon2]:
        if codon1[diff] in purine and codon2[diff] in purine:
            return [weight, 0, 0, 0]
        elif codon1[diff] in pyrimidine and codon2[diff] in pyrimidine:
            return [weight, 0, 0, 0]
        else:
            return [0, weight, 0, 0]
    elif codon1[diff] in purine and codon2[diff] in purine:
        return [0, 0, weight, 0]
    elif codon1[diff] in pyrimidine and codon2[diff] in pyrimidine:
        return [0, 0, weight, 0]
    else:
        return [0, 0, 0, weight]