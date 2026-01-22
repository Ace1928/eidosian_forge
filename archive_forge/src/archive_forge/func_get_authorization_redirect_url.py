from __future__ import annotations
import json
import contextlib
from abc import ABC
from urllib.parse import urljoin
from lazyops.libs import lazyload
from fastapi import Request
from fastapi.background import BackgroundTasks
from ..utils.lazy import get_az_settings, get_az_mtg_api, get_az_resource_schema, logger
from ..utils.helpers import get_hashed_key, create_code_challenge, parse_scopes, encode_params_to_url
from typing import Optional, List, Dict, Any, Union, Type
def get_authorization_redirect_url(self, scope: Optional[str]=None, scopes: Optional[List[str]]=None, audience: Optional[str]=None, **kwargs) -> str:
    """
        Returns the Authorization Redirect URL
        """
    scopes = parse_scopes(scope=scope, scopes=scopes)
    assert scopes, 'At least one scope must be provided'
    scope = ' '.join(scopes)
    if audience is None:
        audience = self.settings.audience
    assert audience, 'Audience must be provided'
    params = {'response_type': 'code', 'code_challenge': self.code_challenge, 'code_challenge_method': 'S256', 'client_id': self.client_id, 'redirect_uri': self.redirect_uri, 'scope': scope, 'audience': audience}
    if kwargs:
        for k, v in kwargs.items():
            if k in params and v:
                params[k] = v
    return encode_params_to_url(params, self.authorize_url)