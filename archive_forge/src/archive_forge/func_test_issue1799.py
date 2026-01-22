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
@pytest.mark.issue(1799)
def test_issue1799():
    """Test sentence boundaries are deserialized correctly, even for
    non-projective sentences."""
    heads_deps = numpy.asarray([[1, 397], [4, 436], [2, 426], [1, 402], [0, 8206900633647566924], [18446744073709551615, 440], [18446744073709551614, 442]], dtype='uint64')
    doc = Doc(Vocab(), words='Just what I was looking for .'.split())
    doc.vocab.strings.add('ROOT')
    doc = doc.from_array([HEAD, DEP], heads_deps)
    assert len(list(doc.sents)) == 1