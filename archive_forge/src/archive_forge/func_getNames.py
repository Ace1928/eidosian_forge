from . import handler
from ._exceptions import SAXNotSupportedException, SAXNotRecognizedException
def getNames(self):
    return list(self._attrs.keys())