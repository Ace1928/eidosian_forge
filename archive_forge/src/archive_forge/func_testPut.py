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
def testPut(self):
    self.cache.PutToken(self.key, self.token_1)
    if not IS_WINDOWS:
        self.assertEqual(384, stat.S_IMODE(os.stat(self.cache.CacheFileName(self.key)).st_mode))