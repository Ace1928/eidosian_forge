import pytest
from spacy.tokens import Doc
from spacy.training.example import Example
from spacy.util import to_ternary_int
from spacy.vocab import Vocab
@pytest.mark.parametrize('annots', [{'words': ['ice', 'cream'], 'weirdannots': ['something', 'such']}])
def test_Example_from_dict_invalid(annots):
    vocab = Vocab()
    predicted = Doc(vocab, words=annots['words'])
    with pytest.raises(KeyError):
        Example.from_dict(predicted, annots)