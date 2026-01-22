from __future__ import absolute_import, unicode_literals
import base64
import hashlib
import json
import logging
from oauthlib import common
from .. import errors
from .base import GrantTypeBase
def create_authorization_code(self, request):
    """
        Generates an authorization grant represented as a dictionary.

        :param request: OAuthlib request.
        :type request: oauthlib.common.Request
        """
    grant = {'code': common.generate_token()}
    if hasattr(request, 'state') and request.state:
        grant['state'] = request.state
    log.debug('Created authorization code grant %r for request %r.', grant, request)
    return grant