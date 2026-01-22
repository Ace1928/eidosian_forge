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
@pytest.mark.issue(2564)
def test_issue2564():
    """Test the tagger sets has_annotation("TAG") correctly when used via Language.pipe."""
    nlp = Language()
    tagger = nlp.add_pipe('tagger')
    tagger.add_label('A')
    nlp.initialize()
    doc = nlp('hello world')
    assert doc.has_annotation('TAG')
    docs = nlp.pipe(['hello', 'world'])
    piped_doc = next(docs)
    assert piped_doc.has_annotation('TAG')