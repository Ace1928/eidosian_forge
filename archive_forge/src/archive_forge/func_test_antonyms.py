import unittest
from nltk.corpus import wordnet as wn
from nltk.corpus import wordnet_ic as wnic
def test_antonyms(self):
    self.assertEqual(L('leader.n.1.leader').antonyms(), [L('follower.n.01.follower')])
    self.assertEqual(L('increase.v.1.increase').antonyms(), [L('decrease.v.01.decrease')])