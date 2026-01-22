from __future__ import absolute_import, unicode_literals
import logging
from oauthlib.common import Request
from oauthlib.oauth2.rfc6749 import utils
from .base import BaseEndpoint, catch_errors_and_unavailability
@property
def default_response_type_handler(self):
    return self.response_types.get(self.default_response_type)