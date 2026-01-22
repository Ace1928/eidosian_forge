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
def testIsGCEServerNotFound(self):
    self.mock_http.request.side_effect = httplib2.ServerNotFoundError()
    self.assertFalse(oauth2_client._IsGCE())
    self.mock_http.request.assert_called_once_with(oauth2_client.METADATA_SERVER)