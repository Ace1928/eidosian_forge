import pytest
from spacy.tokens import Doc
from spacy.training.example import Example
from spacy.util import to_ternary_int
from spacy.vocab import Vocab
@pytest.mark.parametrize('annots', [{'words': ['Sarah', "'s", 'sister', 'flew'], 'morphs': ['NounType=prop|Number=sing', 'Poss=yes', 'Number=sing', 'Tense=past|VerbForm=fin']}])
def test_Example_from_dict_with_morphology(annots):
    vocab = Vocab()
    predicted = Doc(vocab, words=annots['words'])
    example = Example.from_dict(predicted, annots)
    for i, token in enumerate(example.reference):
        assert str(token.morph) == annots['morphs'][i]