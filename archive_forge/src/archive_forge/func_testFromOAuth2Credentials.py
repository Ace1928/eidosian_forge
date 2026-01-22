import base64
import datetime
import json
import os
import unittest
import mock
from mock import patch
from six.moves import http_client
from six.moves import urllib
from oauth2client import client
from oauth2client import client
from google_reauth import reauth
from google_reauth import errors
from google_reauth import reauth_creds
from google_reauth import _reauth_client
from google_reauth.reauth_creds import Oauth2WithReauthCredentials
def testFromOAuth2Credentials(self):
    orig = client.OAuth2Credentials(access_token='at', client_id='ci', client_secret='cs', refresh_token='rt', token_expiry='te', token_uri='tu', user_agent='ua')
    cred = Oauth2WithReauthCredentials.from_OAuth2Credentials(orig)
    self.assertEqual('Oauth2WithReauthCredentials', cred.__class__.__name__)
    self.assertEqual('ci', cred.client_id)
    self.assertEqual('cs', cred.client_secret)