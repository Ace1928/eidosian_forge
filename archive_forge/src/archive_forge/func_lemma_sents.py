import os
import re
from functools import reduce
from nltk.corpus.reader import TaggedCorpusReader, concat
from nltk.corpus.reader.xmldocs import XMLCorpusView
def lemma_sents(self, fileids=None):
    """
        :param fileids: A list specifying the fileids that should be used.
        :return: the given file(s) as a list of sentences or utterances, each
                 encoded as a list of tuples of the word and the corresponding
                 lemma (word, lemma)
        :rtype: list(list(tuple(str, str)))
        """
    return concat([MTEFileReader(os.path.join(self._root, f)).lemma_sents() for f in self.__fileids(fileids)])