import manilaclient
from manilaclient import api_versions
from manilaclient.tests.unit.v2 import fake_clients as fakes
from manilaclient.v2 import client
def get_security_services_detail(self, **kw):
    security_services = {'security_services': [{'id': 1111, 'name': 'fake_security_service', 'description': 'fake_description', 'share_network_id': 'fake_share-network_id', 'user': 'fake_user', 'password': 'fake_password', 'domain': 'fake_domain', 'server': 'fake_server', 'dns_ip': 'fake_dns_ip', 'ou': 'fake_ou', 'type': 'fake_type', 'status': 'fake_status', 'project_id': 'fake_project_id', 'updated_at': 'fake_updated_at'}]}
    return (200, {}, security_services)