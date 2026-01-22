import pytest
import logging
import bs4
from bs4 import BeautifulSoup
from bs4.dammit import (
def test_deprecated_override_encodings(self):
    hebrew = b'\xed\xe5\xec\xf9'
    dammit = UnicodeDammit(hebrew, known_definite_encodings=['shift-jis'], override_encodings=['utf-8'], user_encodings=['iso-8859-8'])
    assert 'iso-8859-8' == dammit.original_encoding
    assert ['shift-jis', 'utf-8', 'iso-8859-8'] == [x[0] for x in dammit.tried_encodings]