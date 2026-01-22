import contextlib
import sys
import unittest
from io import StringIO
from nltk.corpus import gutenberg
from nltk.text import Text
def test_concordance_width(self):
    list_out = ['monstrous', 'monstrous', 'monstrous', 'monstrous', 'monstrous', 'monstrous', 'Monstrous', 'monstrous', 'monstrous', 'monstrous', 'monstrous']
    concordance_out = self.text.concordance_list(self.query, width=0)
    self.assertEqual(list_out, [c.query for c in concordance_out])