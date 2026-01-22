import pytest
import srsly
import spacy
from spacy import schemas
from spacy.tokens import Doc, Span, Token
from .test_underscore import clean_underscore  # noqa: F401
def test_doc_to_json_with_token_attributes_missing(doc):
    Token.set_extension('token_test', default=False)
    Span.set_extension('span_test', default=False)
    doc[0:1]._.span_test = 'span_attribute'
    doc[0]._.token_test = 117
    json_doc = doc.to_json(underscore=['span_test'])
    assert 'underscore_span' in json_doc
    assert json_doc['underscore_span']['span_test'][0]['value'] == 'span_attribute'
    assert 'underscore_token' not in json_doc
    assert len(schemas.validate(schemas.DocJSONSchema, json_doc)) == 0