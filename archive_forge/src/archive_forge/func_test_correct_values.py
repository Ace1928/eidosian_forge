import unittest
import nltk.data
from nltk.corpus.reader.util import (
def test_correct_values(self):
    for f, file_data in self.data():
        v = StreamBackedCorpusView(f, read_whitespace_block)
        self.assertEqual(list(v), file_data.split())
        v = StreamBackedCorpusView(f, read_line_block)
        self.assertEqual(list(v), self.linetok.tokenize(file_data))