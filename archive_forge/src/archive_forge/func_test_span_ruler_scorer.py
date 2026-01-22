import pytest
from thinc.api import NumpyOps, get_current_ops
import spacy
from spacy import registry
from spacy.errors import MatchPatternError
from spacy.tests.util import make_tempdir
from spacy.tokens import Span
from spacy.training import Example
def test_span_ruler_scorer(overlapping_patterns):
    nlp = spacy.blank('xx')
    ruler = nlp.add_pipe('span_ruler')
    ruler.add_patterns(overlapping_patterns)
    text = 'foo bar baz'
    pred_doc = ruler(nlp.make_doc(text))
    assert len(pred_doc.spans['ruler']) == 2
    assert pred_doc.spans['ruler'][0].label_ == 'FOOBAR'
    assert pred_doc.spans['ruler'][1].label_ == 'BARBAZ'
    ref_doc = nlp.make_doc(text)
    ref_doc.spans['ruler'] = [Span(ref_doc, 0, 2, label='FOOBAR')]
    scores = nlp.evaluate([Example(pred_doc, ref_doc)])
    assert scores['spans_ruler_p'] == 0.5
    assert scores['spans_ruler_r'] == 1.0