import pytest
from thinc.api import NumpyOps, get_current_ops
import spacy
from spacy import registry
from spacy.errors import MatchPatternError
from spacy.tests.util import make_tempdir
from spacy.tokens import Span
from spacy.training import Example
def test_span_ruler_ents_bad_filter(overlapping_patterns):

    @registry.misc('test_pass_through_filter')
    def make_pass_through_filter():

        def pass_through_filter(spans1, spans2):
            return spans1 + spans2
        return pass_through_filter
    nlp = spacy.blank('xx')
    ruler = nlp.add_pipe('span_ruler', config={'annotate_ents': True, 'ents_filter': {'@misc': 'test_pass_through_filter'}})
    ruler.add_patterns(overlapping_patterns)
    with pytest.raises(ValueError):
        ruler(nlp.make_doc('foo bar baz'))