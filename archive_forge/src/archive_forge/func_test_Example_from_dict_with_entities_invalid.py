import pytest
from spacy.tokens import Doc
from spacy.training.example import Example
from spacy.util import to_ternary_int
from spacy.vocab import Vocab
@pytest.mark.parametrize('annots', [{'words': ['I', 'like', 'New', 'York', 'and', 'Berlin', '.'], 'entities': [(0, 4, 'LOC'), (21, 27, 'LOC')]}])
def test_Example_from_dict_with_entities_invalid(annots):
    vocab = Vocab()
    predicted = Doc(vocab, words=annots['words'])
    with pytest.warns(UserWarning):
        example = Example.from_dict(predicted, annots)
    assert len(list(example.reference.ents)) == 0