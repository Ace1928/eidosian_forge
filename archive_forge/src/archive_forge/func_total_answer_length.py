from ..pari import pari
import string
from itertools import combinations, combinations_with_replacement, product
def total_answer_length(I):
    return sum([len(list(p)) for p in I.gens()])