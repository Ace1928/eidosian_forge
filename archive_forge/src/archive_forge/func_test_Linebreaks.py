from __future__ import annotations
from importlib import reload
from io import BytesIO
from typing_extensions import Literal
from twisted.trial.unittest import TestCase
from twisted.web import domhelpers, microdom, sux
def test_Linebreaks(self) -> None:
    s = '<test \na="a"\n\tb="#b" />'
    d = microdom.parseString(s)
    e = d.documentElement
    self.assertEqual(e.getAttribute('a'), 'a')
    self.assertEqual(e.getAttribute('b'), '#b')