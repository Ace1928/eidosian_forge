import pickle
import re
import pytest
from spacy.lang.en import English
from spacy.lang.it import Italian
from spacy.language import Language
from spacy.tokenizer import Tokenizer
from spacy.training import Example
from spacy.util import load_config_from_str
from ..util import make_tempdir
@pytest.mark.issue(2482)
def test_issue2482():
    """Test we can serialize and deserialize a blank NER or parser model."""
    nlp = Italian()
    nlp.add_pipe('ner')
    b = nlp.to_bytes()
    Italian().from_bytes(b)