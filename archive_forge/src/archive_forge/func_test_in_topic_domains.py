import unittest
from nltk.corpus import wordnet as wn
from nltk.corpus import wordnet_ic as wnic
def test_in_topic_domains(self):
    self.assertEqual(S('computer_science.n.01').in_topic_domains()[0], S('access.n.05'))
    self.assertEqual(S('germany.n.01').in_region_domains()[23], S('trillion.n.02'))
    self.assertEqual(S('slang.n.02').in_usage_domains()[1], S('airhead.n.01'))