from unittest import mock
from oslo_serialization import jsonutils
import sys
from keystoneauth1 import fixture
import requests
def get_configuration(self):
    return {'auth': {'username': USERNAME, 'password': PASSWORD, 'token': AUTH_TOKEN}, 'region': REGION_NAME, 'identity_api_version': VERSION}