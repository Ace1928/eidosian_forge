import logging
import os
import tempfile
import unittest
import numpy as np
from gensim.models.poincare import PoincareRelations, PoincareModel, PoincareKeyedVectors
from gensim.test.utils import datapath
def test_encoding_handling(self):
    """Tests whether utf8 and non-utf8 data loaded correctly."""
    non_utf8_file = datapath('poincare_cp852.tsv')
    relations = [relation for relation in PoincareRelations(non_utf8_file, encoding='cp852')]
    self.assertEqual(len(relations), 2)
    self.assertEqual(relations[0], (u'tímto', u'budeš'))
    utf8_file = datapath('poincare_utf8.tsv')
    relations = [relation for relation in PoincareRelations(utf8_file)]
    self.assertEqual(len(relations), 2)
    self.assertEqual(relations[0], (u'tímto', u'budeš'))