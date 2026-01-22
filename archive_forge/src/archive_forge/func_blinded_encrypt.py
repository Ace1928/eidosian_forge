import logging
import warnings
from rsa._compat import range
import rsa.prime
import rsa.pem
import rsa.common
import rsa.randnum
import rsa.core
def blinded_encrypt(self, message):
    """Encrypts the message using blinding to prevent side-channel attacks.

        :param message: the message to encrypt
        :type message: int

        :returns: the encrypted message
        :rtype: int
        """
    blind_r = self._get_blinding_factor()
    blinded = self.blind(message, blind_r)
    encrypted = rsa.core.encrypt_int(blinded, self.d, self.n)
    return self.unblind(encrypted, blind_r)