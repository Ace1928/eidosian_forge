import unittest
from nltk.corpus import wordnet as wn
from nltk.corpus import wordnet_ic as wnic
def test_iterable_type_for_all_lemma_names(self):
    cat_lemmas = wn.all_lemma_names(lang='cat')
    eng_lemmas = wn.all_lemma_names(lang='eng')
    self.assertTrue(hasattr(eng_lemmas, '__iter__'))
    self.assertTrue(hasattr(eng_lemmas, '__next__') or hasattr(eng_lemmas, 'next'))
    self.assertTrue(eng_lemmas.__iter__() is eng_lemmas)
    self.assertTrue(hasattr(cat_lemmas, '__iter__'))
    self.assertTrue(hasattr(cat_lemmas, '__next__') or hasattr(eng_lemmas, 'next'))
    self.assertTrue(cat_lemmas.__iter__() is cat_lemmas)