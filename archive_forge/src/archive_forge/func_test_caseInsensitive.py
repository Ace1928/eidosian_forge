from __future__ import annotations
from importlib import reload
from io import BytesIO
from typing_extensions import Literal
from twisted.trial.unittest import TestCase
from twisted.web import domhelpers, microdom, sux
def test_caseInsensitive(self) -> None:
    s = "<foo a='b'><BAx>x</bax></FOO>"
    s2 = '<foo a="b"><bax>x</bax></foo>'
    s3 = "<FOO a='b'><BAx>x</BAx></FOO>"
    s4 = "<foo A='b'>x</foo>"
    d = microdom.parseString(s)
    d2 = microdom.parseString(s2)
    d3 = microdom.parseString(s3, caseInsensitive=1)
    d4 = microdom.parseString(s4, caseInsensitive=1, preserveCase=1)
    d5 = microdom.parseString(s4, caseInsensitive=1, preserveCase=0)
    d6 = microdom.parseString(s4, caseInsensitive=0, preserveCase=0)
    out = microdom.parseString(s).documentElement.toxml()
    self.assertRaises(microdom.MismatchedTags, microdom.parseString, s, caseInsensitive=0)
    self.assertEqual(out, s2)
    self.assertTrue(d.isEqualToDocument(d2))
    self.assertTrue(d.isEqualToDocument(d3))
    self.assertTrue(d4.documentElement.hasAttribute('a'))
    self.assertFalse(d6.documentElement.hasAttribute('a'))
    self.assertEqual(d4.documentElement.toxml(), '<foo A="b">x</foo>')
    self.assertEqual(d5.documentElement.toxml(), '<foo a="b">x</foo>')