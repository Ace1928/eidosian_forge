import pytest
from mock import Mock
from spacy.tokens import Doc, Span, Token
from spacy.tokens.underscore import Underscore
def test_underscore_docstring(en_vocab):
    """Test that docstrings are available for extension methods, even though
    they're partials."""

    def test_method(doc, arg1=1, arg2=2):
        """I am a docstring"""
        return (arg1, arg2)
    Doc.set_extension('test_docstrings', method=test_method)
    doc = Doc(en_vocab, words=['hello', 'world'])
    assert test_method.__doc__ == 'I am a docstring'
    assert doc._.test_docstrings.__doc__.rsplit('. ')[-1] == 'I am a docstring'