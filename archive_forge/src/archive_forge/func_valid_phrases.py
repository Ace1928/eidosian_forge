import warnings
from collections import defaultdict
from math import log
@staticmethod
def valid_phrases(all_phrases_from, hypothesis):
    """
        Extract phrases from ``all_phrases_from`` that contains words
        that have not been translated by ``hypothesis``

        :param all_phrases_from: Phrases represented by their spans, in
            the same format as the return value of
            ``find_all_src_phrases``
        :type all_phrases_from: list(list(int))

        :type hypothesis: _Hypothesis

        :return: A list of phrases, represented by their spans, that
            cover untranslated positions.
        :rtype: list(tuple(int, int))
        """
    untranslated_spans = hypothesis.untranslated_spans(len(all_phrases_from))
    valid_phrases = []
    for available_span in untranslated_spans:
        start = available_span[0]
        available_end = available_span[1]
        while start < available_end:
            for phrase_end in all_phrases_from[start]:
                if phrase_end > available_end:
                    break
                valid_phrases.append((start, phrase_end))
            start += 1
    return valid_phrases