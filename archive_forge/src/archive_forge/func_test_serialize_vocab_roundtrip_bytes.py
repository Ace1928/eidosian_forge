import pickle
import pytest
from thinc.api import get_current_ops
import spacy
from spacy.lang.en import English
from spacy.strings import StringStore
from spacy.tokens import Doc
from spacy.util import ensure_path, load_model
from spacy.vectors import Vectors
from spacy.vocab import Vocab
from ..util import make_tempdir
@pytest.mark.parametrize('strings1,strings2', test_strings)
def test_serialize_vocab_roundtrip_bytes(strings1, strings2):
    vocab1 = Vocab(strings=strings1)
    vocab2 = Vocab(strings=strings2)
    vocab1_b = vocab1.to_bytes()
    vocab2_b = vocab2.to_bytes()
    if strings1 == strings2:
        assert vocab1_b == vocab2_b
    else:
        assert vocab1_b != vocab2_b
    vocab1 = vocab1.from_bytes(vocab1_b)
    assert vocab1.to_bytes() == vocab1_b
    new_vocab1 = Vocab().from_bytes(vocab1_b)
    assert new_vocab1.to_bytes() == vocab1_b
    assert len(new_vocab1.strings) == len(strings1)
    assert sorted([s for s in new_vocab1.strings]) == sorted(strings1)