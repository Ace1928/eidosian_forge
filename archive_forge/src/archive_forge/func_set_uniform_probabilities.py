import warnings
from collections import defaultdict
from math import factorial
from nltk.translate import AlignedSent, Alignment, IBMModel, IBMModel3
from nltk.translate.ibm_model import Counts, longest_target_sentence_length
def set_uniform_probabilities(self, sentence_aligned_corpus):
    """
        Set distortion probabilities uniformly to
        1 / cardinality of displacement values
        """
    max_m = longest_target_sentence_length(sentence_aligned_corpus)
    if max_m <= 1:
        initial_prob = IBMModel.MIN_PROB
    else:
        initial_prob = 1 / (2 * (max_m - 1))
    if initial_prob < IBMModel.MIN_PROB:
        warnings.warn('A target sentence is too long (' + str(max_m) + ' words). Results may be less accurate.')
    for dj in range(1, max_m):
        self.head_distortion_table[dj] = defaultdict(lambda: defaultdict(lambda: initial_prob))
        self.head_distortion_table[-dj] = defaultdict(lambda: defaultdict(lambda: initial_prob))
        self.non_head_distortion_table[dj] = defaultdict(lambda: initial_prob)
        self.non_head_distortion_table[-dj] = defaultdict(lambda: initial_prob)