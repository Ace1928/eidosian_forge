import warnings
from collections import defaultdict
from math import log
def find_all_src_phrases(self, src_sentence):
    """
        Finds all subsequences in src_sentence that have a phrase
        translation in the translation table

        :type src_sentence: tuple(str)

        :return: Subsequences that have a phrase translation,
            represented as a table of lists of end positions.
            For example, if result[2] is [5, 6, 9], then there are
            three phrases starting from position 2 in ``src_sentence``,
            ending at positions 5, 6, and 9 exclusive. The list of
            ending positions are in ascending order.
        :rtype: list(list(int))
        """
    sentence_length = len(src_sentence)
    phrase_indices = [[] for _ in src_sentence]
    for start in range(0, sentence_length):
        for end in range(start + 1, sentence_length + 1):
            potential_phrase = src_sentence[start:end]
            if potential_phrase in self.phrase_table:
                phrase_indices[start].append(end)
    return phrase_indices