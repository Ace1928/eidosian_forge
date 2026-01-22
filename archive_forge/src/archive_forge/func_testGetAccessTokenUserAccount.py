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
def testGetAccessTokenUserAccount(self):
    self.client = CreateMockUserAccountClient(self.mock_datetime)
    self._RunGetAccessTokenTest(expected_rapt=RAPT_TOKEN)