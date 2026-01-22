import warnings
from collections import defaultdict
from math import log
def expansion_score(self, hypothesis, translation_option, src_phrase_span):
    """
        Calculate the score of expanding ``hypothesis`` with
        ``translation_option``

        :param hypothesis: Hypothesis being expanded
        :type hypothesis: _Hypothesis

        :param translation_option: Information about the proposed expansion
        :type translation_option: PhraseTableEntry

        :param src_phrase_span: Word position span of the source phrase
        :type src_phrase_span: tuple(int, int)
        """
    score = hypothesis.raw_score
    score += translation_option.log_prob
    score += self.language_model.probability_change(hypothesis, translation_option.trg_phrase)
    score += self.distortion_score(hypothesis, src_phrase_span)
    score -= self.word_penalty * len(translation_option.trg_phrase)
    return score