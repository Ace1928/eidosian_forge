from . import handler
from ._exceptions import SAXNotSupportedException, SAXNotRecognizedException
def getQNameByName(self, name):
    return self._qnames[name]