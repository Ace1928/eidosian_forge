import pytest
import srsly
import spacy
from spacy import schemas
from spacy.tokens import Doc, Span, Token
from .test_underscore import clean_underscore  # noqa: F401
def test_doc_to_json_span(doc):
    """Test that Doc.to_json() includes spans"""
    doc.spans['test'] = [Span(doc, 0, 2, 'test'), Span(doc, 0, 1, 'test')]
    json_doc = doc.to_json()
    assert 'spans' in json_doc
    assert len(json_doc['spans']) == 1
    assert len(json_doc['spans']['test']) == 2
    assert json_doc['spans']['test'][0]['start'] == 0
    assert len(schemas.validate(schemas.DocJSONSchema, json_doc)) == 0