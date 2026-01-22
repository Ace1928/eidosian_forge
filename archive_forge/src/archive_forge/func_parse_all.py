import itertools
from nltk.internals import overridden
def parse_all(self, sent, *args, **kwargs):
    """:rtype: list(Tree)"""
    return list(self.parse(sent, *args, **kwargs))