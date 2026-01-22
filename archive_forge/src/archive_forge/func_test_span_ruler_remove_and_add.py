import pytest
from thinc.api import NumpyOps, get_current_ops
import spacy
from spacy import registry
from spacy.errors import MatchPatternError
from spacy.tests.util import make_tempdir
from spacy.tokens import Span
from spacy.training import Example
def test_span_ruler_remove_and_add():
    nlp = spacy.blank('xx')
    ruler = nlp.add_pipe('span_ruler')
    patterns1 = [{'label': 'DATE1', 'pattern': 'last time'}]
    ruler.add_patterns(patterns1)
    doc = ruler(nlp.make_doc('I saw him last time we met, this time he brought some flowers'))
    assert len(ruler.patterns) == 1
    assert len(doc.spans['ruler']) == 1
    assert doc.spans['ruler'][0].label_ == 'DATE1'
    assert doc.spans['ruler'][0].text == 'last time'
    patterns2 = [{'label': 'DATE2', 'pattern': 'this time'}]
    ruler.add_patterns(patterns2)
    doc = ruler(nlp.make_doc('I saw him last time we met, this time he brought some flowers'))
    assert len(ruler.patterns) == 2
    assert len(doc.spans['ruler']) == 2
    assert doc.spans['ruler'][0].label_ == 'DATE1'
    assert doc.spans['ruler'][0].text == 'last time'
    assert doc.spans['ruler'][1].label_ == 'DATE2'
    assert doc.spans['ruler'][1].text == 'this time'
    ruler.remove('DATE1')
    doc = ruler(nlp.make_doc('I saw him last time we met, this time he brought some flowers'))
    assert len(ruler.patterns) == 1
    assert len(doc.spans['ruler']) == 1
    assert doc.spans['ruler'][0].label_ == 'DATE2'
    assert doc.spans['ruler'][0].text == 'this time'
    ruler.add_patterns(patterns1)
    doc = ruler(nlp.make_doc('I saw him last time we met, this time he brought some flowers'))
    assert len(ruler.patterns) == 2
    assert len(doc.spans['ruler']) == 2
    patterns3 = [{'label': 'DATE3', 'pattern': 'another time'}]
    ruler.add_patterns(patterns3)
    doc = ruler(nlp.make_doc('I saw him last time we met, this time he brought some flowers, another time some chocolate.'))
    assert len(ruler.patterns) == 3
    assert len(doc.spans['ruler']) == 3
    ruler.remove('DATE3')
    doc = ruler(nlp.make_doc('I saw him last time we met, this time he brought some flowers, another time some chocolate.'))
    assert len(ruler.patterns) == 2
    assert len(doc.spans['ruler']) == 2