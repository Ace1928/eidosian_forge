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
def test_refresh_token(self):
    token_info1 = {'access_token': 'tokentoken', 'token_type': 'Bearer', 'expires_in': 3600}
    new_token1 = self.conn.refresh_token(token_info1)
    self.assertEqual(new_token1['access_token'], STUB_IA_TOKEN['access_token'])
    token_info2 = {'access_token': 'tokentoken', 'token_type': 'Bearer', 'expires_in': 3600, 'refresh_token': 'refreshrefresh'}
    new_token2 = self.conn.refresh_token(token_info2)
    self.assertEqual(new_token2['access_token'], STUB_REFRESH_TOKEN['access_token'])
    self.assertTrue('refresh_token' in new_token1)
    self.assertTrue('refresh_token' in new_token2)