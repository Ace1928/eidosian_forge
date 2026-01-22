import pytest
from spacy.language import Language
from spacy.pipeline.functions import merge_subtokens
from spacy.tokens import Doc, Span
from ..doc.test_underscore import clean_underscore  # noqa: F401
def test_merge_subtokens(doc):
    doc = merge_subtokens(doc)
    assert [t.text for t in doc] == ['This', 'is', 'a sentence', '.', 'This', 'is', 'another sentence', '.', 'And a third .']