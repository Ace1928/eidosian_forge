from pdb import set_trace
import logging
import os
import pickle
import pytest
import sys
import tempfile
from bs4 import (
from bs4.builder import (
from bs4.element import (
from . import (
import warnings
def test_ascii_in_unicode_out(self):
    chardet = dammit.chardet_dammit
    logging.disable(logging.WARNING)
    try:

        def noop(str):
            return None
        dammit.chardet_dammit = noop
        ascii = b'<foo>a</foo>'
        soup_from_ascii = self.soup(ascii)
        unicode_output = soup_from_ascii.decode()
        assert isinstance(unicode_output, str)
        assert unicode_output == self.document_for(ascii.decode())
        assert soup_from_ascii.original_encoding.lower() == 'utf-8'
    finally:
        logging.disable(logging.NOTSET)
        dammit.chardet_dammit = chardet