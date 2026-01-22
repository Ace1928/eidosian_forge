from typing import cast
from zope.interface import Attribute, Interface, implementer
from twisted.web import sux
def gotCData(self, data):
    if self.currElem is not None:
        if isinstance(data, bytes):
            data = data.decode('ascii')
        self.currElem.addContent(data)