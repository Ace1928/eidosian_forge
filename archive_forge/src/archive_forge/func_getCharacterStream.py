from . import handler
from ._exceptions import SAXNotSupportedException, SAXNotRecognizedException
def getCharacterStream(self):
    """Get the character stream for this input source."""
    return self.__charfile