import copy
import struct
from io import BytesIO
from twisted.internet import protocol
from twisted.persisted import styles
from twisted.python import log
from twisted.python.compat import iterbytes
from twisted.python.reflect import fullyQualifiedName
def gotItem(self, item):
    l = self.listStack
    if l:
        l[-1][1].append(item)
    else:
        self.callExpressionReceived(item)