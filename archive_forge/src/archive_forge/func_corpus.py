import re
from collections import defaultdict
from nltk.corpus.reader.util import concat
from nltk.corpus.reader.xmldocs import ElementTree, XMLCorpusReader
from nltk.util import LazyConcatenation, LazyMap, flatten
def corpus(self, fileids=None):
    """
        :return: the given file(s) as a dict of ``(corpus_property_key, value)``
        :rtype: list(dict)
        """
    if not self._lazy:
        return [self._get_corpus(fileid) for fileid in self.abspaths(fileids)]
    return LazyMap(self._get_corpus, self.abspaths(fileids))