import unittest
from nltk.translate.meteor_score import meteor_score
def test_meteor(self):
    score = meteor_score(self.reference, self.candidate, preprocess=str.lower)
    assert score == 0.9921875