import inspect
import keyword
import unittest
from collections import namedtuple
from unittest import mock
from bpython import autocomplete, inspection
from bpython.line import LinePart
def matches_from_completions(self, cursor, line, block, history, completions):
    with mock.patch('bpython.autocomplete.jedi.Script') as Script:
        script = Script.return_value
        script.complete.return_value = completions
        com = autocomplete.MultilineJediCompletion()
        return com.matches(cursor, line, current_block=block, history=history)