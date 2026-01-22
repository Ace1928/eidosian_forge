from sys import maxsize
from nltk.util import trigrams
def guess_language(self, text):
    """Find the language with the min distance
        to the text and return its ISO 639-3 code"""
    self.last_distances = self.lang_dists(text)
    return min(self.last_distances, key=self.last_distances.get)