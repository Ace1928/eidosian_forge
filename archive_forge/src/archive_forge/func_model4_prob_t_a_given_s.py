import warnings
from collections import defaultdict
from math import factorial
from nltk.translate import AlignedSent, Alignment, IBMModel, IBMModel3
from nltk.translate.ibm_model import Counts, longest_target_sentence_length
@staticmethod
def model4_prob_t_a_given_s(alignment_info, ibm_model):
    probability = 1.0
    MIN_PROB = IBMModel.MIN_PROB

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

    def fertility_term():
        value = 1.0
        src_sentence = alignment_info.src_sentence
        for i in range(1, len(src_sentence)):
            fertility = alignment_info.fertility_of_i(i)
            value *= factorial(fertility) * ibm_model.fertility_table[fertility][src_sentence[i]]
            if value < MIN_PROB:
                return MIN_PROB
        return value

    def lexical_translation_term(j):
        t = alignment_info.trg_sentence[j]
        i = alignment_info.alignment[j]
        s = alignment_info.src_sentence[i]
        return ibm_model.translation_table[t][s]

    def distortion_term(j):
        t = alignment_info.trg_sentence[j]
        i = alignment_info.alignment[j]
        if i == 0:
            return 1.0
        if alignment_info.is_head_word(j):
            previous_cept = alignment_info.previous_cept(j)
            src_class = None
            if previous_cept is not None:
                previous_s = alignment_info.src_sentence[previous_cept]
                src_class = ibm_model.src_classes[previous_s]
            trg_class = ibm_model.trg_classes[t]
            dj = j - alignment_info.center_of_cept(previous_cept)
            return ibm_model.head_distortion_table[dj][src_class][trg_class]
        previous_position = alignment_info.previous_in_tablet(j)
        trg_class = ibm_model.trg_classes[t]
        dj = j - previous_position
        return ibm_model.non_head_distortion_table[dj][trg_class]
    probability *= null_generation_term()
    if probability < MIN_PROB:
        return MIN_PROB
    probability *= fertility_term()
    if probability < MIN_PROB:
        return MIN_PROB
    for j in range(1, len(alignment_info.trg_sentence)):
        probability *= lexical_translation_term(j)
        if probability < MIN_PROB:
            return MIN_PROB
        probability *= distortion_term(j)
        if probability < MIN_PROB:
            return MIN_PROB
    return probability