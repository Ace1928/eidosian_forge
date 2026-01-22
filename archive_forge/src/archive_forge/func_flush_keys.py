from .layer_base import AxolotlBaseLayer
from yowsup.layers import YowLayerEvent, EventCallback
from yowsup.layers.network.layer import YowNetworkLayer
from yowsup.layers.axolotl.protocolentities import *
from yowsup.layers.auth.layer_authentication import YowAuthenticationProtocolLayer
from yowsup.layers.protocol_acks.protocolentities import OutgoingAckProtocolEntity
from axolotl.util.hexutil import HexUtil
from axolotl.ecc.curve import Curve
import logging
import binascii
def flush_keys(self, signed_prekey, prekeys, reboot_connection=False):
    """
        sends prekeys
        :return:
        :rtype:
        """
    preKeysDict = {}
    for prekey in prekeys:
        keyPair = prekey.getKeyPair()
        preKeysDict[self.adjustId(prekey.getId())] = self.adjustArray(keyPair.getPublicKey().serialize()[1:])
    signedKeyTuple = (self.adjustId(signed_prekey.getId()), self.adjustArray(signed_prekey.getKeyPair().getPublicKey().serialize()[1:]), self.adjustArray(signed_prekey.getSignature()))
    setKeysIq = SetKeysIqProtocolEntity(self.adjustArray(self.manager.identity.getPublicKey().serialize()[1:]), signedKeyTuple, preKeysDict, Curve.DJB_TYPE, self.adjustId(self.manager.registration_id))
    onResult = lambda _, __: self.on_keys_flushed(prekeys, reboot_connection=reboot_connection)
    self._sendIq(setKeysIq, onResult, self.onSentKeysError)