from __future__ import annotations
from importlib import reload
from io import BytesIO
from typing_extensions import Literal
from twisted.trial.unittest import TestCase
from twisted.web import domhelpers, microdom, sux
def test_replaceNonChild(self) -> None:
    """
        L{Node.replaceChild} raises L{ValueError} if the node given to be
        replaced is not a child of the node C{replaceChild} is called on.
        """
    parent = microdom.parseString('<foo />')
    orphan = microdom.parseString('<bar />')
    replacement = microdom.parseString('<baz />')
    self.assertRaises(ValueError, parent.replaceChild, replacement, orphan)