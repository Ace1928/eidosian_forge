import pytest
from spacy.tokens import Doc
from spacy.training.example import Example
from spacy.util import to_ternary_int
from spacy.vocab import Vocab
def test_Example_init_requires_doc_objects():
    vocab = Vocab()
    with pytest.raises(TypeError):
        Example(None, None)
    with pytest.raises(TypeError):
        Example(Doc(vocab, words=['hi']), None)
    with pytest.raises(TypeError):
        Example(None, Doc(vocab, words=['hi']))