import pytest
import srsly
import spacy
from spacy import schemas
from spacy.tokens import Doc, Span, Token
from .test_underscore import clean_underscore  # noqa: F401
def test_json_to_doc(doc):
    json_doc = doc.to_json()
    json_doc = srsly.json_loads(srsly.json_dumps(json_doc))
    new_doc = Doc(doc.vocab).from_json(json_doc, validate=True)
    assert new_doc.text == doc.text == 'c d e '
    assert len(new_doc) == len(doc) == 3
    assert new_doc[0].pos == doc[0].pos
    assert new_doc[0].tag == doc[0].tag
    assert new_doc[0].dep == doc[0].dep
    assert new_doc[0].head.idx == doc[0].head.idx
    assert new_doc[0].lemma == doc[0].lemma
    assert len(new_doc.ents) == 1
    assert new_doc.ents[0].start == 1
    assert new_doc.ents[0].end == 2
    assert new_doc.ents[0].label_ == 'ORG'
    assert doc.to_bytes() == new_doc.to_bytes()