from __future__ import annotations
from importlib import reload
from io import BytesIO
from typing_extensions import Literal
from twisted.trial.unittest import TestCase
from twisted.web import domhelpers, microdom, sux
def checkParsed(self, input: str, expected: str, beExtremelyLenient: int=1) -> None:
    """
        Check that C{input}, when parsed, produces a DOM where the XML
        of the document element is equal to C{expected}.
        """
    output = microdom.parseString(input, beExtremelyLenient=beExtremelyLenient)
    self.assertEqual(output.documentElement.toxml(), expected)