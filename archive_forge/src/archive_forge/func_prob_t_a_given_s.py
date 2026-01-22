import warnings
from collections import defaultdict
from math import factorial
from nltk.translate import AlignedSent, Alignment, IBMModel, IBMModel3
from nltk.translate.ibm_model import Counts, longest_target_sentence_length
def prob_t_a_given_s(self, alignment_info):
    """
        Probability of target sentence and an alignment given the
        source sentence
        """
    return IBMModel4.model4_prob_t_a_given_s(alignment_info, self)