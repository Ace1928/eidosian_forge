from __future__ import annotations
from importlib import reload
from io import BytesIO
from typing_extensions import Literal
from twisted.trial.unittest import TestCase
from twisted.web import domhelpers, microdom, sux
def test_brokenOpeningTag(self) -> None:
    """
        Check that microdom does its best to handle broken opening tags.
        The important thing is that it doesn't raise an exception.
        """
    input = '<body><h1><sp!\n an>Hello World!</span></h1></body>'
    expected = '<body><h1><sp an="True">Hello World!</sp></h1></body>'
    self.checkParsed(input, expected)