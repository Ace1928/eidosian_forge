import unittest
import os
from weakref import proxy
from functools import partial
from textwrap import dedent
def test_references(self):
    Builder = self.import_builder()
    Builder.load_string(dedent('\n        <TLangClass>:\n            textinput: textinput\n            TLangClass2:\n                id: textinput\n        '))
    wid = TLangClass()
    Builder.apply(wid)
    self.assertTrue(hasattr(wid, 'textinput'))
    self.assertTrue(getattr(wid, 'textinput') is not None)