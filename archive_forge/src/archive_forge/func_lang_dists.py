from sys import maxsize
from nltk.util import trigrams
def lang_dists(self, text):
    """Calculate the "out-of-place" measure between
        the text and all languages"""
    distances = {}
    profile = self.profile(text)
    for lang in self._corpus._all_lang_freq.keys():
        lang_dist = 0
        for trigram in profile:
            lang_dist += self.calc_dist(lang, trigram, profile)
        distances[lang] = lang_dist
    return distances