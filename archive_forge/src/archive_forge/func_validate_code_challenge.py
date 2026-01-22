from __future__ import absolute_import, unicode_literals
import base64
import hashlib
import json
import logging
from oauthlib import common
from .. import errors
from .base import GrantTypeBase
def validate_code_challenge(self, challenge, challenge_method, verifier):
    if challenge_method in self._code_challenge_methods:
        return self._code_challenge_methods[challenge_method](verifier, challenge)
    raise NotImplementedError('Unknown challenge_method %s' % challenge_method)