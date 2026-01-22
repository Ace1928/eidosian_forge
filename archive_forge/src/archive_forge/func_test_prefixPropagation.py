from __future__ import annotations
from importlib import reload
from io import BytesIO
from typing_extensions import Literal
from twisted.trial.unittest import TestCase
from twisted.web import domhelpers, microdom, sux
def test_prefixPropagation(self) -> None:
    """
        Children of prefixed tags respect the default namespace at the point
        where they are rendered.  Specifically, they are not influenced by the
        prefix of their parent as that prefix has no bearing on them.

        See U{http://www.w3.org/TR/xml-names/#scoping} for details.

        To further clarify the matter, the following::

            <root xmlns="http://example.com/ns/test">
                <mytag xmlns="http://example.com/ns/mytags">
                    <mysubtag xmlns="http://example.com/ns/mytags">
                        <element xmlns="http://example.com/ns/test"></element>
                    </mysubtag>
                </mytag>
            </root>

        Should become this after all the namespace declarations have been
        I{moved up}::

            <root xmlns="http://example.com/ns/test"
                  xmlns:mytags="http://example.com/ns/mytags">
                <mytags:mytag>
                    <mytags:mysubtag>
                        <element></element>
                    </mytags:mysubtag>
                </mytags:mytag>
            </root>
        """
    outerNamespace = 'http://example.com/outer'
    innerNamespace = 'http://example.com/inner'
    document = microdom.Document()
    root = document.createElement('root', namespace=outerNamespace)
    document.appendChild(root)
    root.addPrefixes({innerNamespace: 'inner'})
    mytag = document.createElement('mytag', namespace=innerNamespace)
    root.appendChild(mytag)
    mysubtag = document.createElement('mysubtag', namespace=outerNamespace)
    mytag.appendChild(mysubtag)
    xmlOk = '<?xml version="1.0"?><root xmlns="http://example.com/outer" xmlns:inner="http://example.com/inner"><inner:mytag><mysubtag></mysubtag></inner:mytag></root>'
    xmlOut = document.toxml()
    self.assertEqual(xmlOut, xmlOk)