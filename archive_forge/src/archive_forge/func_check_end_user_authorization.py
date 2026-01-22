from __future__ import print_function
import httplib2
import json
import os
from select import select
import stat
from sys import stdin
import time
import webbrowser
from base64 import (
from six.moves.urllib.parse import parse_qs
from lazr.restfulclient.errors import HTTPError
from lazr.restfulclient.authorize.oauth import (
from launchpadlib import uris
def check_end_user_authorization(self, credentials):
    """Check if the end-user authorized"""
    try:
        credentials.exchange_request_token_for_access_token(self.web_root)
    except HTTPError as e:
        if e.response.status == 403:
            raise EndUserDeclinedAuthorization(e.content)
        else:
            if e.response.status != 401:
                print('Unexpected response from Launchpad:')
                print(e)
            raise EndUserNoAuthorization(e.content)
    return credentials.access_token is not None