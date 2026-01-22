from yowsup.common import YowConstants
from yowsup.layers.protocol_iq.protocolentities import IqProtocolEntity
from yowsup.structs import ProtocolTreeNode
import hashlib
import base64
import os
from yowsup.common.tools import WATools
def setRequestArguments(self, mediaType, b64Hash, size, origHash=None):
    assert mediaType in self.__class__.TYPES_MEDIA, 'Expected media type to be in %s, got %s' % (self.__class__.TYPES_MEDIA, mediaType)
    self.mediaType = mediaType
    self.b64Hash = b64Hash
    self.size = int(size)
    self.origHash = origHash