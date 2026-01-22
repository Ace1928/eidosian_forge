import pytest
from mock import Mock
from spacy.tokens import Doc, Span, Token
from spacy.tokens.underscore import Underscore
def test_underscore_dir(en_vocab):
    """Test that dir() correctly returns extension attributes. This enables
    things like tab-completion for the attributes in doc._."""
    Doc.set_extension('test_dir', default=None)
    doc = Doc(en_vocab, words=['hello', 'world'])
    assert '_' in dir(doc)
    assert 'test_dir' in dir(doc._)
    assert 'test_dir' not in dir(doc[0]._)
    assert 'test_dir' not in dir(doc[0:2]._)