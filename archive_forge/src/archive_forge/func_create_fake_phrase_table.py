import unittest
from collections import defaultdict
from math import log
from nltk.translate import PhraseTable, StackDecoder
from nltk.translate.stack_decoder import _Hypothesis, _Stack
@staticmethod
def create_fake_phrase_table():
    phrase_table = PhraseTable()
    phrase_table.add(('hovercraft',), ('',), 0.8)
    phrase_table.add(('my', 'hovercraft'), ('', ''), 0.7)
    phrase_table.add(('my', 'cheese'), ('', ''), 0.7)
    phrase_table.add(('is',), ('',), 0.8)
    phrase_table.add(('is',), ('',), 0.5)
    phrase_table.add(('full', 'of'), ('', ''), 0.01)
    phrase_table.add(('full', 'of', 'eels'), ('', '', ''), 0.5)
    phrase_table.add(('full', 'of', 'spam'), ('', ''), 0.5)
    phrase_table.add(('eels',), ('',), 0.5)
    phrase_table.add(('spam',), ('',), 0.5)
    return phrase_table