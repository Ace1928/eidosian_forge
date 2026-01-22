import pytest
from mock import Mock
from spacy.language import Language
from spacy.pipe_analysis import get_attr_info, validate_attrs
def test_analysis_validate_attrs_valid():
    attrs = ['doc.sents', 'doc.ents', 'token.tag', 'token._.xyz', 'span._.xyz']
    assert validate_attrs(attrs)
    for attr in attrs:
        assert validate_attrs([attr])
    with pytest.raises(ValueError):
        validate_attrs(['doc.sents', 'doc.xyz'])