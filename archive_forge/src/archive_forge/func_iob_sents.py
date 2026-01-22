import textwrap
from nltk.corpus.reader.api import *
from nltk.corpus.reader.util import *
from nltk.tag import map_tag
from nltk.tree import Tree
from nltk.util import LazyConcatenation, LazyMap
def iob_sents(self, fileids=None, tagset=None):
    """
        :return: a list of lists of word/tag/IOB tuples
        :rtype: list(list)
        :param fileids: the list of fileids that make up this corpus
        :type fileids: None or str or list
        """
    self._require(self.WORDS, self.POS, self.CHUNK)

    def get_iob_words(grid):
        return self._get_iob_words(grid, tagset)
    return LazyMap(get_iob_words, self._grids(fileids))