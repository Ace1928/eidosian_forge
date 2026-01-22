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
@pytest.mark.issue(1834)
def test_issue1834():
    """Test that sentence boundaries & parse/tag flags are not lost
    during serialization."""
    words = ['This', 'is', 'a', 'first', 'sentence', '.', 'And', 'another', 'one']
    doc = Doc(Vocab(), words=words)
    doc[6].is_sent_start = True
    new_doc = Doc(doc.vocab).from_bytes(doc.to_bytes())
    assert new_doc[6].sent_start
    assert not new_doc.has_annotation('DEP')
    assert not new_doc.has_annotation('TAG')
    doc = Doc(Vocab(), words=words, tags=['TAG'] * len(words), heads=[0, 0, 0, 0, 0, 0, 6, 6, 6], deps=['dep'] * len(words))
    new_doc = Doc(doc.vocab).from_bytes(doc.to_bytes())
    assert new_doc[6].sent_start
    assert new_doc.has_annotation('DEP')
    assert new_doc.has_annotation('TAG')