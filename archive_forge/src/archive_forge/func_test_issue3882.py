import numpy
import pytest
from spacy import displacy
from spacy.displacy.render import DependencyRenderer, EntityRenderer, SpanRenderer
from spacy.lang.en import English
from spacy.lang.fa import Persian
from spacy.tokens import Doc, Span
@pytest.mark.issue(3882)
def test_issue3882(en_vocab):
    """Test that displaCy doesn't serialize the doc.user_data when making a
    copy of the Doc.
    """
    doc = Doc(en_vocab, words=['Hello', 'world'], deps=['dep', 'dep'])
    doc.user_data['test'] = set()
    displacy.parse_deps(doc)