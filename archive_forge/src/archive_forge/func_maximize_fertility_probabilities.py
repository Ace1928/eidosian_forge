from bisect import insort_left
from collections import defaultdict
from copy import deepcopy
from math import ceil
def maximize_fertility_probabilities(self, counts):
    for phi, src_words in counts.fertility.items():
        for s in src_words:
            estimate = counts.fertility[phi][s] / counts.fertility_for_any_phi[s]
            self.fertility_table[phi][s] = max(estimate, IBMModel.MIN_PROB)