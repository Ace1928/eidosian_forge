from __future__ import annotations
from importlib import reload
from io import BytesIO
from typing_extensions import Literal
from twisted.trial.unittest import TestCase
from twisted.web import domhelpers, microdom, sux
def test_brokenAttributeValue(self) -> None:
    """
        Check that microdom encompasses broken attribute values.
        """
    input = '<body><h1><div align="cen!\n ter">Foo</div></h1></body>'
    expected = '<body><h1><div align="cen!\n ter">Foo</div></h1></body>'
    self.checkParsed(input, expected)