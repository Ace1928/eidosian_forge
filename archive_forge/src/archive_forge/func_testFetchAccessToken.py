from __future__ import absolute_import
import datetime
import logging
import os
import stat
import sys
import unittest
from freezegun import freeze_time
from gcs_oauth2_boto_plugin import oauth2_client
import httplib2
@freeze_time('2014-03-26 01:01:01')
def testFetchAccessToken(self):
    token = 'my_token'
    self.mock_http.request.return_value = (FakeResponse(200), '{"access_token":"%(TOKEN)s", "expires_in": %(EXPIRES_IN)d}' % {'TOKEN': token, 'EXPIRES_IN': 42})
    client = oauth2_client.OAuth2GCEClient()
    self.assertEqual(str(client.FetchAccessToken()), 'AccessToken(token=%s, expiry=2014-03-26 01:01:43Z)' % token)
    self.mock_http.request.assert_called_with(oauth2_client.META_TOKEN_URI, method='GET', body=None, headers=oauth2_client.META_HEADERS)