import re
import sys
import textwrap
from io import StringIO
from .. import commands, export_pot, option, registry, tests
def test_from_string_missing(self):
    """When string has no lineno the old context details are returned"""
    path = 'str_missing.py'
    context = export_pot._ModuleContext(path, 4, ({}, {'line\n': 21}))
    context1 = context.from_string('line\n')
    context2A = context.from_string('not there')
    self.check_context(context2A, path, 4)
    context2B = context1.from_string('not there')
    self.check_context(context2B, path, 21)
    self.assertContainsRe(self.get_log(), "String b?'not there' not found")