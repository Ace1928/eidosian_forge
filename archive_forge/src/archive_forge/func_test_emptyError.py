from __future__ import annotations
from importlib import reload
from io import BytesIO
from typing_extensions import Literal
from twisted.trial.unittest import TestCase
from twisted.web import domhelpers, microdom, sux
def test_emptyError(self) -> None:
    self.assertRaises(sux.ParseError, microdom.parseString, '')