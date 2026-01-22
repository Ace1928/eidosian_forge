from yowsup.common import YowConstants
from yowsup.layers.protocol_iq.protocolentities import IqProtocolEntity
from yowsup.structs import ProtocolTreeNode
import os,  struct
def setProps(self, identityKey, signedPreKey, preKeys, djbType, registrationId=None):
    assert type(preKeys) is dict, 'Expected keys to be a dict key_id -> public_key'
    assert type(signedPreKey) is tuple, 'Exception signed pre key to be tuple id,key,signature'
    self.preKeys = preKeys
    self.identityKey = identityKey
    self.registration = registrationId or os.urandom(4)
    self.djbType = int(djbType)
    self.signedPreKey = signedPreKey