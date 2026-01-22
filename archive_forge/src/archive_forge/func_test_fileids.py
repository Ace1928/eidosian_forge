import unittest
import pytest
from nltk.corpus import (  # mwa_ppdb
from nltk.tree import Tree
def test_fileids(self):
    self.assertEqual(mwa_ppdb.fileids(), ['ppdb-1.0-xxxl-lexical.extended.synonyms.uniquepairs'])