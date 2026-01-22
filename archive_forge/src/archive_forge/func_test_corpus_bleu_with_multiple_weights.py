import io
import unittest
from nltk.data import find
from nltk.translate.bleu_score import (
def test_corpus_bleu_with_multiple_weights(self):
    hyp1 = ['It', 'is', 'a', 'guide', 'to', 'action', 'which', 'ensures', 'that', 'the', 'military', 'always', 'obeys', 'the', 'commands', 'of', 'the', 'party']
    ref1a = ['It', 'is', 'a', 'guide', 'to', 'action', 'that', 'ensures', 'that', 'the', 'military', 'will', 'forever', 'heed', 'Party', 'commands']
    ref1b = ['It', 'is', 'the', 'guiding', 'principle', 'which', 'guarantees', 'the', 'military', 'forces', 'always', 'being', 'under', 'the', 'command', 'of', 'the', 'Party']
    ref1c = ['It', 'is', 'the', 'practical', 'guide', 'for', 'the', 'army', 'always', 'to', 'heed', 'the', 'directions', 'of', 'the', 'party']
    hyp2 = ['he', 'read', 'the', 'book', 'because', 'he', 'was', 'interested', 'in', 'world', 'history']
    ref2a = ['he', 'was', 'interested', 'in', 'world', 'history', 'because', 'he', 'read', 'the', 'book']
    weight_1 = (1, 0, 0, 0)
    weight_2 = (0.25, 0.25, 0.25, 0.25)
    weight_3 = (0, 0, 0, 0, 1)
    bleu_scores = corpus_bleu(list_of_references=[[ref1a, ref1b, ref1c], [ref2a]], hypotheses=[hyp1, hyp2], weights=[weight_1, weight_2, weight_3])
    assert bleu_scores[0] == corpus_bleu([[ref1a, ref1b, ref1c], [ref2a]], [hyp1, hyp2], weight_1)
    assert bleu_scores[1] == corpus_bleu([[ref1a, ref1b, ref1c], [ref2a]], [hyp1, hyp2], weight_2)
    assert bleu_scores[2] == corpus_bleu([[ref1a, ref1b, ref1c], [ref2a]], [hyp1, hyp2], weight_3)