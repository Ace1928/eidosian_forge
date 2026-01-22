from __future__ import absolute_import, unicode_literals
import copy
import json
import logging
from ....common import unicode_type
from .base import BaseEndpoint, catch_errors_and_unavailability
from .authorization import AuthorizationEndpoint
from .introspect import IntrospectEndpoint
from .token import TokenEndpoint
from .revocation import RevocationEndpoint
from .. import grant_types
def validate_metadata_authorization(self, claims, endpoint):
    claims.setdefault('response_types_supported', list(filter(lambda x: x != 'none', endpoint._response_types.keys())))
    claims.setdefault('response_modes_supported', ['query', 'fragment'])
    if 'token' in claims['response_types_supported']:
        self._grant_types.append('implicit')
    self.validate_metadata(claims, 'response_types_supported', is_required=True, is_list=True)
    self.validate_metadata(claims, 'response_modes_supported', is_list=True)
    if 'code' in claims['response_types_supported']:
        code_grant = endpoint._response_types['code']
        if not isinstance(code_grant, grant_types.AuthorizationCodeGrant) and hasattr(code_grant, 'default_grant'):
            code_grant = code_grant.default_grant
        claims.setdefault('code_challenge_methods_supported', list(code_grant._code_challenge_methods.keys()))
        self.validate_metadata(claims, 'code_challenge_methods_supported', is_list=True)
    self.validate_metadata(claims, 'authorization_endpoint', is_required=True, is_url=True)