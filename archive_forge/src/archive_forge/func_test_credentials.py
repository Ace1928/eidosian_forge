import contextlib
import json
import sys
from StringIO import StringIO
import traceback
from google.appengine.api import app_identity
import google.auth
from google.auth import _helpers
from google.auth import app_engine
import google.auth.transport.urllib3
import urllib3.contrib.appengine
import webapp2
def test_credentials():
    credentials = app_engine.Credentials()
    scoped_credentials = credentials.with_scopes([EMAIL_SCOPE])
    scoped_credentials.refresh(None)
    assert scoped_credentials.valid
    assert scoped_credentials.token is not None
    url = _helpers.update_query(TOKEN_INFO_URL, {'access_token': scoped_credentials.token})
    response = HTTP_REQUEST(url=url, method='GET')
    token_info = json.loads(response.data.decode('utf-8'))
    assert token_info['scope'] == EMAIL_SCOPE