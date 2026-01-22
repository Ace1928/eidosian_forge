import pytest
from thinc.api import NumpyOps, get_current_ops
import spacy
from spacy import registry
from spacy.errors import MatchPatternError
from spacy.tests.util import make_tempdir
from spacy.tokens import Span
from spacy.training import Example
def test_span_ruler_remove_nonexisting_pattern(person_org_patterns):
    nlp = spacy.blank('xx')
    ruler = nlp.add_pipe('span_ruler')
    ruler.add_patterns(person_org_patterns)
    assert len(ruler.patterns) == 3
    with pytest.raises(ValueError):
        ruler.remove('NE')
    with pytest.raises(ValueError):
        ruler.remove_by_id('NE')