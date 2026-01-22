from functools import total_ordering
from xml.etree import ElementTree
from nltk.corpus.reader.api import *
from nltk.corpus.reader.util import *
from nltk.internals import raise_unorderable_types
from nltk.tree import Tree
def nouns(self):
    """
        :return: a corpus view that acts as a list of all noun lemmas
            in this corpus (from the nombank.1.0.words file).
        """
    return StreamBackedCorpusView(self.abspath(self._nounsfile), read_line_block, encoding=self.encoding(self._nounsfile))