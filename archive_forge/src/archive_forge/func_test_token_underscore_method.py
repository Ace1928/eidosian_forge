import pytest
from mock import Mock
from spacy.tokens import Doc, Span, Token
from spacy.tokens.underscore import Underscore
def test_token_underscore_method():
    token = Mock(doc=Mock(), idx=7, say_cheese=lambda token: 'cheese')
    Underscore.token_extensions['hello'] = (None, token.say_cheese, None, None)
    token._ = Underscore(Underscore.token_extensions, token, start=token.idx)
    assert token._.hello() == 'cheese'