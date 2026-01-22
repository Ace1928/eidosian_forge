from dissononce.processing.impl.cipherstate import CipherState
from dissononce.processing.symmetricstate import SymmetricState as BaseSymmetricState
def get_handshake_hash(self):
    """
        GetHandshakeHash():
        Returns h. This function should only be called at the end of a handshake,
        i.e. after the Split() function has been called

        :return: h
        :rtype: bytes
        """
    return self._h