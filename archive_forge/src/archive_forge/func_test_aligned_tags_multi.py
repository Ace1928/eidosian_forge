import pytest
from spacy.tokens import Doc
from spacy.training.example import Example
from spacy.util import to_ternary_int
from spacy.vocab import Vocab
def test_aligned_tags_multi():
    pred_words = ['Applysome', 'sunscreen', 'unless', 'you', 'can', 'not']
    gold_words = ['Apply', 'somesun', 'screen', 'unless', 'you', 'cannot']
    gold_tags = ['VERB', 'DET', 'NOUN', 'SCONJ', 'PRON', 'VERB']
    annots = {'words': gold_words, 'tags': gold_tags}
    vocab = Vocab()
    predicted = Doc(vocab, words=pred_words)
    example = Example.from_dict(predicted, annots)
    aligned_tags = example.get_aligned('TAG', as_string=True)
    assert aligned_tags == [None, None, 'SCONJ', 'PRON', 'VERB', 'VERB']