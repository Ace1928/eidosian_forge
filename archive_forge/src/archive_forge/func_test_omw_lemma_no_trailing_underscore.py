import unittest
from nltk.corpus import wordnet as wn
from nltk.corpus import wordnet_ic as wnic
def test_omw_lemma_no_trailing_underscore(self):
    expected = sorted(['popolna_sprememba_v_mišljenju', 'popoln_obrat', 'preobrat', 'preobrat_v_mišljenju'])
    self.assertEqual(sorted(S('about-face.n.02').lemma_names(lang='slv')), expected)