import re
import numpy
import pytest
from spacy.lang.de import German
from spacy.lang.en import English
from spacy.symbols import ORTH
from spacy.tokenizer import Tokenizer
from spacy.tokens import Doc
from spacy.training import Example
from spacy.util import (
from spacy.vocab import Vocab
@pytest.mark.issue(1375)
def test_issue1375():
    """Test that token.nbor() raises IndexError for out-of-bounds access."""
    doc = Doc(Vocab(), words=['0', '1', '2'])
    with pytest.raises(IndexError):
        assert doc[0].nbor(-1)
    assert doc[1].nbor(-1).text == '0'
    with pytest.raises(IndexError):
        assert doc[2].nbor(1)
    assert doc[1].nbor(1).text == '2'