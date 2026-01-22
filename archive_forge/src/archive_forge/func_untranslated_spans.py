import warnings
from collections import defaultdict
from math import log
def untranslated_spans(self, sentence_length):
    """
        Starting from each untranslated word, find the longest
        continuous span of untranslated positions

        :param sentence_length: Length of source sentence being
            translated by the hypothesis
        :type sentence_length: int

        :rtype: list(tuple(int, int))
        """
    translated_positions = self.translated_positions()
    translated_positions.sort()
    translated_positions.append(sentence_length)
    untranslated_spans = []
    start = 0
    for end in translated_positions:
        if start < end:
            untranslated_spans.append((start, end))
        start = end + 1
    return untranslated_spans