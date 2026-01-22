from . import handler
from ._exceptions import SAXNotSupportedException, SAXNotRecognizedException
def setEncoding(self, encoding):
    """Sets the character encoding of this InputSource.

        The encoding must be a string acceptable for an XML encoding
        declaration (see section 4.3.3 of the XML recommendation).

        The encoding attribute of the InputSource is ignored if the
        InputSource also contains a character stream."""
    self.__encoding = encoding