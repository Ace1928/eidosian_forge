import unittest
import os
from weakref import proxy
from functools import partial
from textwrap import dedent
def test_apply_rules(self):
    Builder = self.import_builder()
    Builder.load_string('<TLangClassCustom>:\n\tobj: 42')
    wid = TLangClass()
    Builder.apply(wid)
    self.assertIsNone(wid.obj)
    Builder.apply_rules(wid, 'TLangClassCustom')
    self.assertEqual(wid.obj, 42)