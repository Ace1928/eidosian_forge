from __future__ import annotations
from importlib import reload
from io import BytesIO
from typing_extensions import Literal
from twisted.trial.unittest import TestCase
from twisted.web import domhelpers, microdom, sux
def test_noTags(self) -> None:
    """
        A string with nothing that looks like a tag at all should just
        be parsed as body text.
        """
    s = 'Hi orders!'
    d = microdom.parseString(s, beExtremelyLenient=True)
    self.assertEqual(d.firstChild().toxml(), '<html>Hi orders!</html>')