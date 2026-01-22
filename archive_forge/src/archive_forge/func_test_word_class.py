from weakref import proxy
import copy
import pickle
import regex
import string
import sys
import unittest
def test_word_class(self):
    self.assertEqual(regex.findall('\\w+', ' हिन्दी,'), ['हिन्दी'])
    self.assertEqual(regex.findall('\\W+', ' हिन्दी,'), [' ', ','])
    self.assertEqual(regex.split('(?V1)\\b', ' हिन्दी,'), [' ', 'हिन्दी', ','])
    self.assertEqual(regex.split('(?V1)\\B', ' हिन्दी,'), ['', ' ह', 'ि', 'न', '्', 'द', 'ी,', ''])