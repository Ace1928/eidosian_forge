import warnings
from collections import defaultdict
from math import factorial
from nltk.translate import AlignedSent, Alignment, IBMModel, IBMModel3
from nltk.translate.ibm_model import Counts, longest_target_sentence_length
def lexical_translation_term(j):
    t = alignment_info.trg_sentence[j]
    i = alignment_info.alignment[j]
    s = alignment_info.src_sentence[i]
    return ibm_model.translation_table[t][s]