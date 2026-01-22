import pytest
from spacy.tokens import Doc
from spacy.training.example import Example
from spacy.util import to_ternary_int
from spacy.vocab import Vocab
@pytest.mark.filterwarnings('ignore::UserWarning')
def test_aligned_tags():
    pred_words = ['Apply', 'some', 'sunscreen', 'unless', 'you', 'can', 'not']
    gold_words = ['Apply', 'some', 'sun', 'screen', 'unless', 'you', 'cannot']
    gold_tags = ['VERB', 'DET', 'NOUN', 'NOUN', 'SCONJ', 'PRON', 'VERB']
    annots = {'words': gold_words, 'tags': gold_tags}
    vocab = Vocab()
    predicted = Doc(vocab, words=pred_words)
    example1 = Example.from_dict(predicted, annots)
    aligned_tags1 = example1.get_aligned('TAG', as_string=True)
    assert aligned_tags1 == ['VERB', 'DET', 'NOUN', 'SCONJ', 'PRON', 'VERB', 'VERB']
    example2 = Example.from_dict(predicted, example1.to_dict())
    aligned_tags2 = example2.get_aligned('TAG', as_string=True)
    assert aligned_tags2 == ['VERB', 'DET', 'NOUN', 'SCONJ', 'PRON', 'VERB', 'VERB']