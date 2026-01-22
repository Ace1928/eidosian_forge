import inspect
import keyword
import unittest
from collections import namedtuple
from unittest import mock
from bpython import autocomplete, inspection
from bpython.line import LinePart
def completer(self, matches):
    mock_completer = autocomplete.BaseCompletionType()
    mock_completer.matches = mock.Mock(return_value=matches)
    return mock_completer