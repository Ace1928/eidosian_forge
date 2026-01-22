import unittest
import os
from weakref import proxy
from functools import partial
from textwrap import dedent
def test_invalid_indentation(self):
    Builder = self.import_builder()
    from kivy.lang import ParserException
    kv_code = dedent("            BoxLayout:\n                orientation: 'vertical'\n                    Button:\n        ")
    try:
        Builder.load_string(kv_code)
        self.fail('Invalid indentation.')
    except ParserException:
        pass