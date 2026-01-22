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
def test_custom_builder_class(self):

    class Mock(object):

        def __init__(self, **kwargs):
            self.called_with = kwargs
            self.is_xml = True
            self.store_line_numbers = False
            self.cdata_list_attributes = []
            self.preserve_whitespace_tags = []
            self.string_containers = {}

        def initialize_soup(self, soup):
            pass

        def feed(self, markup):
            self.fed = markup

        def reset(self):
            pass

        def ignore(self, ignore):
            pass
        set_up_substitutions = can_be_empty_element = ignore

        def prepare_markup(self, *args, **kwargs):
            yield ('prepared markup', 'original encoding', 'declared encoding', 'contains replacement characters')
    kwargs = dict(var='value', convertEntities=True)
    with warnings.catch_warnings(record=True):
        soup = BeautifulSoup('', builder=Mock, **kwargs)
    assert isinstance(soup.builder, Mock)
    assert dict(var='value') == soup.builder.called_with
    assert 'prepared markup' == soup.builder.fed
    builder = Mock(**kwargs)
    with warnings.catch_warnings(record=True) as w:
        soup = BeautifulSoup('', builder=builder, ignored_value=True)
    msg = str(w[0].message)
    assert msg.startswith('Keyword arguments to the BeautifulSoup constructor will be ignored.')
    assert builder == soup.builder
    assert kwargs == builder.called_with