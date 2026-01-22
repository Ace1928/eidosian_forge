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
def test_parser_markup_rejection(self):

    class Mock(TreeBuilder):

        def feed(self, *args, **kwargs):
            raise ParserRejectedMarkup('Nope.')

    def prepare_markup(self, *args, **kwargs):
        yield (markup, None, None, False)
        yield (markup, None, None, False)
    import re
    with pytest.raises(ParserRejectedMarkup) as exc_info:
        BeautifulSoup('', builder=Mock)
    assert 'The markup you provided was rejected by the parser. Trying a different parser or a different encoding may help.' in str(exc_info.value)