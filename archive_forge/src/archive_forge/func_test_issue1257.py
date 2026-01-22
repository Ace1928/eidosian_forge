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
@pytest.mark.issue(1257)
def test_issue1257():
    """Test that tokens compare correctly."""
    doc1 = Doc(Vocab(), words=['a', 'b', 'c'])
    doc2 = Doc(Vocab(), words=['a', 'c', 'e'])
    assert doc1[0] != doc2[0]
    assert not doc1[0] == doc2[0]