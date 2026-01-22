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
def testPutGet(self):
    """Tests putting and getting various tokens."""
    self.assertEqual(None, self.cache.GetToken(self.key))
    self.cache.PutToken(self.key, self.token_1)
    cached_token = self.cache.GetToken(self.key)
    self.assertEqual(self.token_1, cached_token)
    self.cache.PutToken(self.key, self.token_2)
    cached_token = self.cache.GetToken(self.key)
    self.assertEqual(self.token_2, cached_token)