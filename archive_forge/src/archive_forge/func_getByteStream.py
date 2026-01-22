from . import handler
from ._exceptions import SAXNotSupportedException, SAXNotRecognizedException
def getByteStream(self):
    """Get the byte stream for this input source.

        The getEncoding method will return the character encoding for
        this byte stream, or None if unknown."""
    return self.__bytefile