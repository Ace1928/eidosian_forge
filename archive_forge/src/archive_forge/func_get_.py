import manilaclient
from manilaclient import api_versions
from manilaclient.tests.unit.v2 import fake_clients as fakes
from manilaclient.v2 import client
def get_(self, **kw):
    body = {'versions': [{'status': 'CURRENT', 'updated': '2015-07-30T11:33:21Z', 'links': [{'href': 'http://docs.openstack.org/', 'type': 'text/html', 'rel': 'describedby'}, {'href': 'http://localhost:8786/v2/', 'rel': 'self'}], 'min_version': '2.0', 'version': self.default_headers['X-Openstack-Manila-Api-Version'], 'id': 'v2.0'}]}
    return (200, {}, body)