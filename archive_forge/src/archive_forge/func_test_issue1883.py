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
@pytest.mark.issue(1883)
def test_issue1883():
    matcher = Matcher(Vocab())
    matcher.add('pat1', [[{'orth': 'hello'}]])
    doc = Doc(matcher.vocab, words=['hello'])
    assert len(matcher(doc)) == 1
    new_matcher = copy.deepcopy(matcher)
    new_doc = Doc(new_matcher.vocab, words=['hello'])
    assert len(new_matcher(new_doc)) == 1