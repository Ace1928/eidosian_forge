import os
import re
from functools import reduce
from nltk.corpus.reader import TaggedCorpusReader, concat
from nltk.corpus.reader.xmldocs import XMLCorpusView
def lemma_words(self, fileids=None):
    """
        :param fileids: A list specifying the fileids that should be used.
        :return: the given file(s) as a list of words, the corresponding lemmas
                 and punctuation symbols, encoded as tuples (word, lemma)
        :rtype: list(tuple(str,str))
        """
    return concat([MTEFileReader(os.path.join(self._root, f)).lemma_words() for f in self.__fileids(fileids)])