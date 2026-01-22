import http.client as http
import urllib.parse as urlparse
import httplib2
from keystoneclient import service_catalog as ks_service_catalog
from oslo_serialization import jsonutils
from glance.common import exception
from glance.i18n import _
def check_auth_params(self):
    for required in ('username', 'password', 'auth_url', 'strategy'):
        if self.creds.get(required) is None:
            raise exception.MissingCredentialError(required=required)
    if self.creds['strategy'] != 'keystone':
        raise exception.BadAuthStrategy(expected='keystone', received=self.creds['strategy'])
    if self.creds['auth_url'].rstrip('/').endswith('v2.0'):
        if self.creds.get('tenant') is None:
            raise exception.MissingCredentialError(required='tenant')
    if self.creds['auth_url'].rstrip('/').endswith('v3'):
        if self.creds.get('project') is None:
            raise exception.MissingCredentialError(required='project')