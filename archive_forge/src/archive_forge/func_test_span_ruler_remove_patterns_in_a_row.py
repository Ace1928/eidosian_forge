import pytest
from thinc.api import NumpyOps, get_current_ops
import spacy
from spacy import registry
from spacy.errors import MatchPatternError
from spacy.tests.util import make_tempdir
from spacy.tokens import Span
from spacy.training import Example
def test_span_ruler_remove_patterns_in_a_row(person_org_date_patterns):
    nlp = spacy.blank('xx')
    ruler = nlp.add_pipe('span_ruler')
    ruler.add_patterns(person_org_date_patterns)
    doc = ruler(nlp.make_doc('Dina founded the company ACME on June 14th'))
    assert len(doc.spans['ruler']) == 3
    assert doc.spans['ruler'][0].label_ == 'PERSON'
    assert doc.spans['ruler'][0].text == 'Dina'
    assert doc.spans['ruler'][1].label_ == 'ORG'
    assert doc.spans['ruler'][1].text == 'ACME'
    assert doc.spans['ruler'][2].label_ == 'DATE'
    assert doc.spans['ruler'][2].text == 'June 14th'
    ruler.remove('ORG')
    ruler.remove('DATE')
    doc = ruler(nlp.make_doc('Dina went to school'))
    assert len(doc.spans['ruler']) == 1