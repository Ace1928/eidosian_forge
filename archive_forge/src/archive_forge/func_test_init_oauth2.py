import os
import sys
import time
import random
import urllib
import datetime
import unittest
import threading
from unittest import mock
import requests
from libcloud.test import MockHttp, LibcloudTestCase
from libcloud.utils.py3 import httplib
from libcloud.common.google import (
def test_init_oauth2(self):
    kwargs = {'auth_type': GoogleAuthType.IA}
    cred = GoogleOAuth2Credential(*GCE_PARAMS, **kwargs)
    self.assertEqual(cred.token, STUB_TOKEN_FROM_FILE)
    with mock.patch.object(GoogleOAuth2Credential, '_get_token_from_file', return_value=None):
        cred = GoogleOAuth2Credential(*GCE_PARAMS, **kwargs)
        expected = STUB_IA_TOKEN
        expected['expire_time'] = cred.token['expire_time']
        self.assertEqual(cred.token, expected)
        cred._write_token_to_file.assert_called_once_with()