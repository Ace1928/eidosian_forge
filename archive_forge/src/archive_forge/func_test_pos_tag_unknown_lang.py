import unittest
from nltk import pos_tag, word_tokenize
def test_pos_tag_unknown_lang(self):
    text = '모르겠 습니 다'
    self.assertRaises(NotImplementedError, pos_tag, word_tokenize(text), lang='kor')
    self.assertRaises(NotImplementedError, pos_tag, word_tokenize(text), lang=None)