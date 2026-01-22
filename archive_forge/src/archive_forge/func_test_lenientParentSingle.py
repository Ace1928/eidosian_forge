from __future__ import annotations
from importlib import reload
from io import BytesIO
from typing_extensions import Literal
from twisted.trial.unittest import TestCase
from twisted.web import domhelpers, microdom, sux
def test_lenientParentSingle(self) -> None:
    """
        Test that the C{parentNode} attribute is set to a meaningful value
        when we parse an HTML document that has a non-Element root node.
        """
    s = 'Hello'
    d = microdom.parseString(s, beExtremelyLenient=1)
    self.assertIdentical(d.documentElement, d.documentElement.firstChild().parentNode)