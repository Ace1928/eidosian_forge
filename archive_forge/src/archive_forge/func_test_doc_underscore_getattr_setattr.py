import pytest
from mock import Mock
from spacy.tokens import Doc, Span, Token
from spacy.tokens.underscore import Underscore
def test_doc_underscore_getattr_setattr():
    doc = Mock()
    doc.doc = doc
    doc.user_data = {}
    Underscore.doc_extensions['hello'] = (False, None, None, None)
    doc._ = Underscore(Underscore.doc_extensions, doc)
    assert doc._.hello is False
    doc._.hello = True
    assert doc._.hello is True