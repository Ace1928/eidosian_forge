import json
from unittest import mock
import uuid
import requests
from cinderclient import client
from cinderclient import exceptions
from cinderclient.tests.unit import utils
def test_get_no_auth_url(self):
    client.HTTPClient('username', 'password', 'project_id', retries=0)