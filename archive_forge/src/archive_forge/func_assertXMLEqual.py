from io import BytesIO
from xml.dom import minidom as dom
from twisted.internet.protocol import FileWrapper
def assertXMLEqual(self, first, second):
    """
        Verify that two strings represent the same XML document.

        @param first: An XML string.
        @type first: L{bytes}

        @param second: An XML string that should match C{first}.
        @type second: L{bytes}
        """
    self.assertEqual(dom.parseString(first).toxml(), dom.parseString(second).toxml())