import os
import re
from functools import reduce
from nltk.corpus.reader import TaggedCorpusReader, concat
from nltk.corpus.reader.xmldocs import XMLCorpusView
def lemma_paras(self, fileids=None):
    """
        :param fileids: A list specifying the fileids that should be used.
        :return: the given file(s) as a list of paragraphs, each encoded as a
                 list of sentences, which are in turn encoded as a list of
                 tuples of the word and the corresponding lemma (word, lemma)
        :rtype: list(List(List(tuple(str, str))))
        """
    return concat([MTEFileReader(os.path.join(self._root, f)).lemma_paras() for f in self.__fileids(fileids)])