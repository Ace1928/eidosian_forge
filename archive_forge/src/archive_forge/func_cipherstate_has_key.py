from dissononce.processing.impl.cipherstate import CipherState
from dissononce.processing.symmetricstate import SymmetricState as BaseSymmetricState
def cipherstate_has_key(self):
    return self._cipherstate.has_key()