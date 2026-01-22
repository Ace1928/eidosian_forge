from __future__ import annotations
from importlib import reload
from io import BytesIO
from typing_extensions import Literal
from twisted.trial.unittest import TestCase
from twisted.web import domhelpers, microdom, sux
def test_namespaceDelete(self) -> None:
    """
        Test that C{toxml} can support xml structures that remove namespaces.
        """
    s1 = '<?xml version="1.0"?><html xmlns="http://www.w3.org/TR/REC-html40"><body xmlns=""></body></html>'
    s2 = microdom.parseString(s1).toxml()
    self.assertEqual(s1, s2)