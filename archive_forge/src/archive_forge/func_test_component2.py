import pytest
from mock import Mock
from spacy.language import Language
from spacy.pipe_analysis import get_attr_info, validate_attrs
@Language.component('c2', requires=['token.tag', 'token.pos'], assigns=['token.lemma', 'doc.tensor'])
def test_component2(doc):
    return doc