import codecs
import os.path
import nltk
from nltk.chunk import tagstr2tree
from nltk.corpus.reader.api import *
from nltk.corpus.reader.bracket_parse import BracketParseCorpusReader
from nltk.corpus.reader.util import *
from nltk.tokenize import *
from nltk.tree import Tree
def chunked_paras(self, fileids=None, tagset=None):
    """
        :return: the given file(s) as a list of
            paragraphs, each encoded as a list of sentences, which are
            in turn encoded as a shallow Tree.  The leaves of these
            trees are encoded as ``(word, tag)`` tuples (if the corpus
            has tags) or word strings (if the corpus has no tags).
        :rtype: list(list(Tree))
        """
    return concat([ChunkedCorpusView(f, enc, 1, 1, 1, 1, *self._cv_args, target_tagset=tagset) for f, enc in self.abspaths(fileids, True)])