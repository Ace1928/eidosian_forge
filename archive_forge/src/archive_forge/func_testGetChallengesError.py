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
def testGetChallengesError(self):
    self.getChallengesError(json.dumps({'status': 'ERROR'}))
    self.getChallengesError(json.dumps({'error': {'message': 'Some error'}}))