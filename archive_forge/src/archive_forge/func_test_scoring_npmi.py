import logging
import unittest
import numpy as np
from gensim.models.phrases import Phrases, FrozenPhrases, _PhrasesTransformation
from gensim.models.phrases import original_scorer
from gensim.test.utils import common_texts, temporary_file, datapath
def test_scoring_npmi(self):
    """Test normalized pointwise mutual information scoring."""
    bigram = Phrases(self.sentences, min_count=1, threshold=0.5, scoring='npmi', connector_words=self.connector_words)
    test_sentences = [['data', 'and', 'graph', 'survey', 'for', 'human', 'interface']]
    seen_scores = set((round(score, 3) for score in bigram.find_phrases(test_sentences).values()))
    assert seen_scores == set([0.74, 0.894])