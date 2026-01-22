from __future__ import annotations
from importlib import reload
from io import BytesIO
from typing_extensions import Literal
from twisted.trial.unittest import TestCase
from twisted.web import domhelpers, microdom, sux
def test_namespaceInheritance(self) -> None:
    """
        Check that unspecified namespace is a thing separate from undefined
        namespace. This test added after discovering some weirdness in Lore.
        """
    child = microdom.Element('ol')
    parent = microdom.Element('div', namespace='http://www.w3.org/1999/xhtml')
    parent.childNodes = [child]
    self.assertEqual(parent.toxml(), '<div xmlns="http://www.w3.org/1999/xhtml"><ol></ol></div>')