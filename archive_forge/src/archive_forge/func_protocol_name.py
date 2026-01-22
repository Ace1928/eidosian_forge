from dissononce.processing.symmetricstate import SymmetricState
from dissononce.processing.handshakestate import HandshakeState as BaseHandshakeState
from dissononce.dh.public import PublicKey
from dissononce.dh.keypair import KeyPair
from dissononce.dh.dh import DH
import logging
@property
def protocol_name(self):
    return self._protocol_name