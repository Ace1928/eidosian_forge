from __future__ import annotations
from importlib import reload
from io import BytesIO
from typing_extensions import Literal
from twisted.trial.unittest import TestCase
from twisted.web import domhelpers, microdom, sux
def test_insensitiveLenient(self) -> None:
    d = microdom.parseString("<?xml version='1.0'?><bar><xA><y>c</Xa> <foo></bar>", beExtremelyLenient=1)
    self.assertEqual(d.documentElement.firstChild().toxml(), '<xa><y>c</y></xa>')