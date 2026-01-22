import unittest
import pytest
from nltk.corpus import (  # mwa_ppdb
from nltk.tree import Tree
def test_esp(self):
    words = cess_esp.words()[:15]
    txt = 'El grupo estatal Electricité_de_France -Fpa- EDF -Fpt- anunció hoy , jueves , la compra del'
    self.assertEqual(words, txt.split())
    self.assertEqual(cess_esp.words()[115], 'años')