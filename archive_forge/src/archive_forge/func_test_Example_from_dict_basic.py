import pytest
from spacy.tokens import Doc
from spacy.training.example import Example
from spacy.util import to_ternary_int
from spacy.vocab import Vocab
def test_Example_from_dict_basic():
    example = Example.from_dict(Doc(Vocab(), words=['hello', 'world']), {'words': ['hello', 'world']})
    assert isinstance(example.x, Doc)
    assert isinstance(example.y, Doc)