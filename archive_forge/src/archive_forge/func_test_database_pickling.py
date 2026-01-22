from __future__ import absolute_import, unicode_literals
import pickle
from abc import ABCMeta, abstractmethod
from copy import deepcopy
from io import BytesIO, TextIOWrapper
import six
import pytest
from pybtex.database import parse_bytes, parse_string, BibliographyData, Entry
from pybtex.plugin import find_plugin
from .data import reference_data
@pytest.mark.parametrize(['protocol'], [(protocol,) for protocol in range(0, pickle.HIGHEST_PROTOCOL + 1)])
def test_database_pickling(protocol):
    check_database_io(PickleIO(protocol))