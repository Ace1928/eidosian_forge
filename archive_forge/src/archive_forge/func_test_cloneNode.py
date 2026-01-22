from __future__ import annotations
from importlib import reload
from io import BytesIO
from typing_extensions import Literal
from twisted.trial.unittest import TestCase
from twisted.web import domhelpers, microdom, sux
def test_cloneNode(self) -> None:
    s = '<foo a="b"><bax>x</bax></foo>'
    node = microdom.parseString(s).documentElement
    clone = node.cloneNode(deep=1)
    self.failIfEquals(node, clone)
    self.assertEqual(len(node.childNodes), len(clone.childNodes))
    c1, c2 = (node.firstChild(), clone.firstChild())
    self.failIfEquals(c1, c2)
    self.assertEqual(len(c1.childNodes), len(c2.childNodes))
    self.failIfEquals(c1.firstChild(), c2.firstChild())
    self.assertEqual(s, clone.toxml())
    self.assertEqual(node.namespace, clone.namespace)