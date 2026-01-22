import warnings
from collections import defaultdict
from math import factorial
from nltk.translate import AlignedSent, Alignment, IBMModel, IBMModel3
from nltk.translate.ibm_model import Counts, longest_target_sentence_length
def null_generation_term():
    value = 1.0
    p1 = ibm_model.p1
    p0 = 1 - p1
    null_fertility = alignment_info.fertility_of_i(0)
    m = len(alignment_info.trg_sentence) - 1
    value *= pow(p1, null_fertility) * pow(p0, m - 2 * null_fertility)
    if value < MIN_PROB:
        return MIN_PROB
    for i in range(1, null_fertility + 1):
        value *= (m - null_fertility - i + 1) / i
    return value