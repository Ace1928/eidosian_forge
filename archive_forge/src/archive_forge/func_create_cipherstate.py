from dissononce.dh.dh import DH
from dissononce.cipher.cipher import Cipher
from dissononce.hash.hash import Hash
from dissononce.processing.handshakepatterns.handshakepattern import HandshakePattern
from dissononce.processing.impl.handshakestate import HandshakeState
from dissononce.processing.impl.symmetricstate import SymmetricState
from dissononce.processing.impl.cipherstate import CipherState
def create_cipherstate(self, cipher=None):
    """
        :param cipher:
        :type cipher: Cipher
        :return:
        :rtype: CipherState
        """
    return CipherState(cipher or self._cipher)