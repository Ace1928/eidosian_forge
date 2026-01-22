import json
import unittest
from six.moves import http_client
from six.moves import urllib
import oauth2client
from oauth2client import client
from oauth2client import transport
from oauth2client.contrib import gce
def test_application_default(self):
    default_creds = client.GoogleCredentials.get_application_default()
    self.assertIsInstance(default_creds, gce.AppAssertionCredentials)