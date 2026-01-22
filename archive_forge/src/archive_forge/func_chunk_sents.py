from nltk.corpus.reader.api import *
from nltk.corpus.reader.xmldocs import XMLCorpusReader, XMLCorpusView
from nltk.tree import Tree
def chunk_sents(self, fileids=None):
    """
        :return: the given file(s) as a list of sentences, each encoded
            as a list of chunks.
        :rtype: list(list(list(str)))
        """
    return self._items(fileids, 'chunk', True, False, False)