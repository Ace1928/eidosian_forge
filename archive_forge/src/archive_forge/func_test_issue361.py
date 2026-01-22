import numpy
import pytest
from spacy.attrs import IS_ALPHA, IS_DIGIT
from spacy.lookups import Lookups
from spacy.tokens import Doc
from spacy.util import OOV_RANK
from spacy.vocab import Vocab
@pytest.mark.issue(361)
@pytest.mark.parametrize('text1,text2', [('cat', 'dog')])
def test_issue361(en_vocab, text1, text2):
    """Test Issue #361: Equality of lexemes"""
    assert en_vocab[text1] == en_vocab[text1]
    assert en_vocab[text1] != en_vocab[text2]