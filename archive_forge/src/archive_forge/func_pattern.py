from dissononce.dh.dh import DH
from dissononce.cipher.cipher import Cipher
from dissononce.hash.hash import Hash
from dissononce.processing.handshakepatterns.handshakepattern import HandshakePattern
from dissononce.processing.impl.handshakestate import HandshakeState
from dissononce.processing.impl.symmetricstate import SymmetricState
from dissononce.processing.impl.cipherstate import CipherState
@property
def pattern(self):
    return self._pattern