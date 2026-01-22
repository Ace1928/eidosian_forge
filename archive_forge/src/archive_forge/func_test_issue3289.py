import copy
import pickle
import numpy
import pytest
from spacy.attrs import DEP, HEAD
from spacy.lang.en import English
from spacy.language import Language
from spacy.matcher import Matcher, PhraseMatcher
from spacy.tokens import Doc
from spacy.vectors import Vectors
from spacy.vocab import Vocab
from ..util import make_tempdir
@pytest.mark.issue(3289)
def test_issue3289():
    """Test that Language.to_bytes handles serializing a pipeline component
    with an uninitialized model."""
    nlp = English()
    nlp.add_pipe('textcat')
    bytes_data = nlp.to_bytes()
    new_nlp = English()
    new_nlp.add_pipe('textcat')
    new_nlp.from_bytes(bytes_data)