import warnings
from collections import defaultdict
from nltk.translate import AlignedSent, Alignment, IBMModel, IBMModel1
from nltk.translate.ibm_model import Counts
def maximize_alignment_probabilities(self, counts):
    MIN_PROB = IBMModel.MIN_PROB
    for i, j_s in counts.alignment.items():
        for j, src_sentence_lengths in j_s.items():
            for l, trg_sentence_lengths in src_sentence_lengths.items():
                for m in trg_sentence_lengths:
                    estimate = counts.alignment[i][j][l][m] / counts.alignment_for_any_i[j][l][m]
                    self.alignment_table[i][j][l][m] = max(estimate, MIN_PROB)