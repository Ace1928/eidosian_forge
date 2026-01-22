import base64
import json
import os
import unittest
import mock
from six.moves import http_client
from six.moves import urllib
from oauth2client import client
from google_reauth import challenges
from google_reauth import reauth
from google_reauth import errors
from google_reauth import reauth_creds
from google_reauth import _reauth_client
from google_reauth.reauth_creds import Oauth2WithReauthCredentials
from pyu2f import model
from pyu2f import u2f
def reauthAccessTokenError(self, response, content):

    def side_effect(*args, **kwargs):
        qp = dict(urllib.parse.parse_qsl(kwargs['body']))
        try:
            qp_json = json.loads(kwargs['body'])
        except ValueError:
            qp_json = {}
        uri = kwargs['uri'] if 'uri' in kwargs else args[0]
        if uri == self.oauth_api_url and qp.get('scope') == reauth._REAUTH_SCOPE:
            return (response, content)
        raise Exception('Unexpected call :/\nURL {0}\n{1}'.format(uri, kwargs['body']))
    with mock.patch('httplib2.Http.request', side_effect=side_effect) as request_mock:
        with self.assertRaises(errors.ReauthAccessTokenRefreshError):
            reauth.get_rapt_token(request_mock, self.client_id, self.client_secret, 'some_refresh_token', self.oauth_api_url)
        self.assertEqual(1, request_mock.call_count)