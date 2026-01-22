import logging
import warnings
from rsa._compat import range
import rsa.prime
import rsa.pem
import rsa.common
import rsa.randnum
import rsa.core
def blinded_decrypt(self, encrypted):
    """Decrypts the message using blinding to prevent side-channel attacks.

        :param encrypted: the encrypted message
        :type encrypted: int

        :returns: the decrypted message
        :rtype: int
        """
    blind_r = self._get_blinding_factor()
    blinded = self.blind(encrypted, blind_r)
    decrypted = rsa.core.decrypt_int(blinded, self.d, self.n)
    return self.unblind(decrypted, blind_r)