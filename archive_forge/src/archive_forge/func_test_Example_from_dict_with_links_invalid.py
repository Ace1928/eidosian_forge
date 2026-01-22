import pytest
from spacy.tokens import Doc
from spacy.training.example import Example
from spacy.util import to_ternary_int
from spacy.vocab import Vocab
@pytest.mark.parametrize('annots', [{'words': ['I', 'like', 'New', 'York', 'and', 'Berlin', '.'], 'links': {(7, 14): {'Q7381115': 1.0, 'Q2146908': 0.0}}}])
def test_Example_from_dict_with_links_invalid(annots):
    vocab = Vocab()
    predicted = Doc(vocab, words=annots['words'])
    with pytest.raises(ValueError):
        Example.from_dict(predicted, annots)