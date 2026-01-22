import pytest
import logging
import bs4
from bs4 import BeautifulSoup
from bs4.dammit import (
def test_encoding_detector_replaces_junk_in_encoding_name_with_replacement_character(self):
    detected = EncodingDetector(b'<?xml version="1.0" encoding="UTF-\xdb" ?>')
    encodings = list(detected.encodings)
    assert 'utf-�' in encodings