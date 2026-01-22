import itertools
from nltk.internals import overridden
def parse_sents(self, sents, *args, **kwargs):
    """
        Apply ``self.parse()`` to each element of ``sents``.
        :rtype: iter(iter(Tree))
        """
    return (self.parse(sent, *args, **kwargs) for sent in sents)