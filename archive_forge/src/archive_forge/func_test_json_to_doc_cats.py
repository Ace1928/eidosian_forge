import pytest
import srsly
import spacy
from spacy import schemas
from spacy.tokens import Doc, Span, Token
from .test_underscore import clean_underscore  # noqa: F401
def test_json_to_doc_cats(doc):
    """Test that Doc.from_json() includes correct .cats."""
    cats = {'A': 0.3, 'B': 0.7}
    doc.cats = cats
    json_doc = doc.to_json()
    new_doc = Doc(doc.vocab).from_json(json_doc, validate=True)
    assert new_doc.cats == cats