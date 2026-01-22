import os
import pytest
from spacy.attrs import IS_ALPHA, LEMMA, ORTH
from spacy.lang.en import English
from spacy.parts_of_speech import NOUN, VERB
from spacy.vocab import Vocab
from ..util import make_tempdir
@pytest.mark.issue(1868)
def test_issue1868():
    """Test Vocab.__contains__ works with int keys."""
    vocab = Vocab()
    lex = vocab['hello']
    assert lex.orth in vocab
    assert lex.orth_ in vocab
    assert 'some string' not in vocab
    int_id = vocab.strings.add('some string')
    assert int_id not in vocab