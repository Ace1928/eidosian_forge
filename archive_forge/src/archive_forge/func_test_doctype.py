from __future__ import annotations
from importlib import reload
from io import BytesIO
from typing_extensions import Literal
from twisted.trial.unittest import TestCase
from twisted.web import domhelpers, microdom, sux
def test_doctype(self) -> None:
    s = '<?xml version="1.0"?><!DOCTYPE foo PUBLIC "baz" "http://www.example.com/example.dtd"><foo></foo>'
    s2 = '<foo/>'
    d = microdom.parseString(s)
    d2 = microdom.parseString(s2)
    self.assertEqual(d.doctype, 'foo PUBLIC "baz" "http://www.example.com/example.dtd"')
    self.assertEqual(d.toxml(), s)
    self.assertFalse(d.isEqualToDocument(d2))
    self.assertTrue(d.documentElement.isEqualToNode(d2.documentElement))