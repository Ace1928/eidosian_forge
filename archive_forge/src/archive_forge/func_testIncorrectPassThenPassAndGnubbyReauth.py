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
def testIncorrectPassThenPassAndGnubbyReauth(self):
    with mock.patch('httplib2.Http.request', side_effect=self._request_mock_side_effect) as request_mock:
        self.getpass_mock.return_value = None
        self.getpass_mock.side_effect = ['bad_pass', self.correct_password]
        reauth_result = self._call_reauth(request_mock)
        self.assertEqual(self.rapt_token, reauth_result)
        self.assertEqual(5, request_mock.call_count)