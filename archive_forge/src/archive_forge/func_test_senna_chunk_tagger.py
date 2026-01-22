import unittest
from os import environ, path, sep
from nltk.classify import Senna
from nltk.tag import SennaChunkTagger, SennaNERTagger, SennaTagger
def test_senna_chunk_tagger(self):
    chktagger = SennaChunkTagger(SENNA_EXECUTABLE_PATH)
    result_1 = chktagger.tag('What is the airspeed of an unladen swallow ?'.split())
    expected_1 = [('What', 'B-NP'), ('is', 'B-VP'), ('the', 'B-NP'), ('airspeed', 'I-NP'), ('of', 'B-PP'), ('an', 'B-NP'), ('unladen', 'I-NP'), ('swallow', 'I-NP'), ('?', 'O')]
    result_2 = list(chktagger.bio_to_chunks(result_1, chunk_type='NP'))
    expected_2 = [('What', '0'), ('the airspeed', '2-3'), ('an unladen swallow', '5-6-7')]
    self.assertEqual(result_1, expected_1)
    self.assertEqual(result_2, expected_2)