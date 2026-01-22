import pytest
from spacy.language import Language
from spacy.pipeline.functions import merge_subtokens
from spacy.tokens import Doc, Span
from ..doc.test_underscore import clean_underscore  # noqa: F401
def test_factories_merge_noun_chunks(doc2):
    assert len(doc2) == 7
    nlp = Language()
    merge_noun_chunks = nlp.create_pipe('merge_noun_chunks')
    merge_noun_chunks(doc2)
    assert len(doc2) == 6
    assert doc2[2].text == 'New York'