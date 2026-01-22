import unittest
from os import environ, path, sep
from nltk.classify import Senna
from nltk.tag import SennaChunkTagger, SennaNERTagger, SennaTagger
def test_senna_ner_tagger(self):
    nertagger = SennaNERTagger(SENNA_EXECUTABLE_PATH)
    result_1 = nertagger.tag('Shakespeare theatre was in London .'.split())
    expected_1 = [('Shakespeare', 'B-PER'), ('theatre', 'O'), ('was', 'O'), ('in', 'O'), ('London', 'B-LOC'), ('.', 'O')]
    result_2 = nertagger.tag('UN headquarters are in NY , USA .'.split())
    expected_2 = [('UN', 'B-ORG'), ('headquarters', 'O'), ('are', 'O'), ('in', 'O'), ('NY', 'B-LOC'), (',', 'O'), ('USA', 'B-LOC'), ('.', 'O')]
    self.assertEqual(result_1, expected_1)
    self.assertEqual(result_2, expected_2)