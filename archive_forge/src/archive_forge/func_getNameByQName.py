from . import handler
from ._exceptions import SAXNotSupportedException, SAXNotRecognizedException
def getNameByQName(self, name):
    for nsname, qname in self._qnames.items():
        if qname == name:
            return nsname
    raise KeyError(name)