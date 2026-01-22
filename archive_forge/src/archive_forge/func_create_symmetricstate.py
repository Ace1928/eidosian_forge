from dissononce.dh.dh import DH
from dissononce.cipher.cipher import Cipher
from dissononce.hash.hash import Hash
from dissononce.processing.handshakepatterns.handshakepattern import HandshakePattern
from dissononce.processing.impl.handshakestate import HandshakeState
from dissononce.processing.impl.symmetricstate import SymmetricState
from dissononce.processing.impl.cipherstate import CipherState
def create_symmetricstate(self, cipherstate=None, hash=None):
    """
        :param cipherstate:
        :type cipherstate: CipherState
        :param hash:
        :type hash: Hash
        :return:
        :rtype: SymmetricState
        """
    return SymmetricState(cipherstate or self.create_cipherstate(), hash or self._hash)