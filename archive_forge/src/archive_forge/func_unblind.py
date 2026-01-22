import logging
import warnings
from rsa._compat import range
import rsa.prime
import rsa.pem
import rsa.common
import rsa.randnum
import rsa.core
def unblind(self, blinded, r):
    """Performs blinding on the message using random number 'r'.

        :param blinded: the blinded message, as integer, to unblind.
        :param r: the random number to unblind with.
        :return: the original message.

        The blinding is such that message = unblind(decrypt(blind(encrypt(message))).

        See https://en.wikipedia.org/wiki/Blinding_%28cryptography%29
        """
    return rsa.common.inverse(r, self.n) * blinded % self.n