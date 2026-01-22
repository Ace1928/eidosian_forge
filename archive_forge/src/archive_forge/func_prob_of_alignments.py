from bisect import insort_left
from collections import defaultdict
from copy import deepcopy
from math import ceil
def prob_of_alignments(self, alignments):
    probability = 0
    for alignment_info in alignments:
        probability += self.prob_t_a_given_s(alignment_info)
    return probability