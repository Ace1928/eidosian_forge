import unittest
import pytest
from nltk.corpus import (  # mwa_ppdb
from nltk.tree import Tree
def test_parsed_sents(self):
    parsed_sents = conll2007.parsed_sents('esp.train')[0]
    self.assertEqual(parsed_sents.tree(), Tree('fortaleció', [Tree('aumento', ['El', Tree('del', [Tree('índice', [Tree('de', [Tree('desempleo', ['estadounidense'])])])])]), 'hoy', 'considerablemente', Tree('al', [Tree('euro', [Tree('cotizaba', [',', 'que', Tree('a', [Tree('15.35', ['las', 'GMT'])]), 'se', Tree('en', [Tree('mercado', ['el', Tree('de', ['divisas']), Tree('de', ['Fráncfort'])])]), Tree('a', ['0,9452_dólares']), Tree('frente_a', [',', Tree('0,9349_dólares', ['los', Tree('de', [Tree('mañana', ['esta'])])])])])])]), '.']))