from __future__ import annotations
from importlib import reload
from io import BytesIO
from typing_extensions import Literal
from twisted.trial.unittest import TestCase
from twisted.web import domhelpers, microdom, sux
def test_prefixedTags(self) -> None:
    """
        XML elements with a prefixed name as per upper level tag definition
        have a start-tag of C{"<prefix:tag>"} and an end-tag of
        C{"</prefix:tag>"}.

        Refer to U{http://www.w3.org/TR/xml-names/#ns-using} for details.
        """
    outerNamespace = 'http://example.com/outer'
    innerNamespace = 'http://example.com/inner'
    document = microdom.Document()
    root = document.createElement('root', namespace=outerNamespace)
    root.addPrefixes({innerNamespace: 'inner'})
    tag = document.createElement('tag', namespace=innerNamespace)
    child = document.createElement('child', namespace=innerNamespace)
    tag.appendChild(child)
    root.appendChild(tag)
    document.appendChild(root)
    xmlOk = '<?xml version="1.0"?><root xmlns="http://example.com/outer" xmlns:inner="http://example.com/inner"><inner:tag><inner:child></inner:child></inner:tag></root>'
    xmlOut = document.toxml()
    self.assertEqual(xmlOut, xmlOk)