import collections
import copy
import datetime
import json
import logging
import os
import shutil
import socket
import sys
import tempfile
import six
from six.moves import http_client
from six.moves import urllib
import oauth2client
from oauth2client import _helpers
from oauth2client import _pkce
from oauth2client import clientsecrets
from oauth2client import transport
@_helpers.positional(2)
def step2_exchange(self, code=None, http=None, device_flow_info=None):
    """Exchanges a code for OAuth2Credentials.

        Args:
            code: string, a dict-like object, or None. For a non-device
                  flow, this is either the response code as a string, or a
                  dictionary of query parameters to the redirect_uri. For a
                  device flow, this should be None.
            http: httplib2.Http, optional http instance to use when fetching
                  credentials.
            device_flow_info: DeviceFlowInfo, return value from step1 in the
                              case of a device flow.

        Returns:
            An OAuth2Credentials object that can be used to authorize requests.

        Raises:
            FlowExchangeError: if a problem occurred exchanging the code for a
                               refresh_token.
            ValueError: if code and device_flow_info are both provided or both
                        missing.
        """
    if code is None and device_flow_info is None:
        raise ValueError('No code or device_flow_info provided.')
    if code is not None and device_flow_info is not None:
        raise ValueError('Cannot provide both code and device_flow_info.')
    if code is None:
        code = device_flow_info.device_code
    elif not isinstance(code, (six.string_types, six.binary_type)):
        if 'code' not in code:
            raise FlowExchangeError(code.get('error', 'No code was supplied in the query parameters.'))
        code = code['code']
    post_data = {'client_id': self.client_id, 'code': code, 'scope': self.scope}
    if self.client_secret is not None:
        post_data['client_secret'] = self.client_secret
    if self._pkce:
        post_data['code_verifier'] = self.code_verifier
    if device_flow_info is not None:
        post_data['grant_type'] = 'http://oauth.net/grant_type/device/1.0'
    else:
        post_data['grant_type'] = 'authorization_code'
        post_data['redirect_uri'] = self.redirect_uri
    body = urllib.parse.urlencode(post_data)
    headers = {'content-type': 'application/x-www-form-urlencoded'}
    if self.authorization_header is not None:
        headers['Authorization'] = self.authorization_header
    if self.user_agent is not None:
        headers['user-agent'] = self.user_agent
    if http is None:
        http = transport.get_http_object()
    resp, content = transport.request(http, self.token_uri, method='POST', body=body, headers=headers)
    d = _parse_exchange_token_response(content)
    if resp.status == http_client.OK and 'access_token' in d:
        access_token = d['access_token']
        refresh_token = d.get('refresh_token', None)
        if not refresh_token:
            logger.info("Received token response with no refresh_token. Consider reauthenticating with prompt='consent'.")
        token_expiry = None
        if 'expires_in' in d:
            delta = datetime.timedelta(seconds=int(d['expires_in']))
            token_expiry = delta + _UTCNOW()
        extracted_id_token = None
        id_token_jwt = None
        if 'id_token' in d:
            extracted_id_token = _extract_id_token(d['id_token'])
            id_token_jwt = d['id_token']
        logger.info('Successfully retrieved access token')
        return OAuth2Credentials(access_token, self.client_id, self.client_secret, refresh_token, token_expiry, self.token_uri, self.user_agent, revoke_uri=self.revoke_uri, id_token=extracted_id_token, id_token_jwt=id_token_jwt, token_response=d, scopes=self.scope, token_info_uri=self.token_info_uri)
    else:
        logger.info('Failed to retrieve access token: %s', content)
        if 'error' in d:
            error_msg = str(d['error']) + str(d.get('error_description', ''))
        else:
            error_msg = 'Invalid response: {0}.'.format(str(resp.status))
        raise FlowExchangeError(error_msg)