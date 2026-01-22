from collections import Counter
from nltk.tokenize.api import TokenizerI
def find_legal_onsets(self, words):
    """
        Gathers all onsets and then return only those above the frequency threshold

        :param words: List of words in a language
        :type words: list(str)
        :return: Set of legal onsets
        :rtype: set(str)
        """
    onsets = [self.onset(word) for word in words]
    legal_onsets = [k for k, v in Counter(onsets).items() if v / len(onsets) > self.legal_frequency_threshold]
    return set(legal_onsets)