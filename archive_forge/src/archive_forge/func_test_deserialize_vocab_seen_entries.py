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
@pytest.mark.parametrize('strings,lex_attr', test_strings_attrs)
def test_deserialize_vocab_seen_entries(strings, lex_attr):
    vocab = Vocab(strings=strings)
    vocab.from_bytes(vocab.to_bytes())
    assert len(vocab.strings) == len(strings)