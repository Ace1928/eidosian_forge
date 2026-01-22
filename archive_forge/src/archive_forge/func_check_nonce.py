from __future__ import absolute_import, unicode_literals
import sys
from . import SIGNATURE_METHODS, utils
def check_nonce(self, nonce):
    """Checks that the nonce only contains only safe characters

        and is no shorter than lower and no longer than upper.
        """
    lower, upper = self.nonce_length
    return set(nonce) <= self.safe_characters and lower <= len(nonce) <= upper