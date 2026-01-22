import re
from nltk.metrics import accuracy as _accuracy
from nltk.tag.mapping import map_tag
from nltk.tag.util import str2tuple
from nltk.tree import Tree
def guessed(self):
    """
        Return the chunks which were included in the guessed
        chunk structures, listed in input order.

        :rtype: list of chunks
        """
    chunks = list(self._guessed)
    return [c[1] for c in chunks]