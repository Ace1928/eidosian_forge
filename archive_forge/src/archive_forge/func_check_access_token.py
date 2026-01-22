from __future__ import absolute_import, unicode_literals
import sys
from . import SIGNATURE_METHODS, utils
def check_access_token(self, request_token):
    """Checks that the token contains only safe characters

        and is no shorter than lower and no longer than upper.
        """
    lower, upper = self.access_token_length
    return set(request_token) <= self.safe_characters and lower <= len(request_token) <= upper