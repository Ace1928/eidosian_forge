import datetime
import hashlib
import json
import uuid
from openstack.cloud import meta
from openstack.orchestration.util import template_format
from openstack import utils
def make_fake_server(server_id, name, status='ACTIVE', admin_pass=None, addresses=None, image=None, flavor=None):
    if addresses is None:
        if status == 'ACTIVE':
            addresses = {'private': [{'OS-EXT-IPS-MAC:mac_addr': 'fa:16:3e:df:b0:8d', 'version': 6, 'addr': 'fddb:b018:307:0:f816:3eff:fedf:b08d', 'OS-EXT-IPS:type': 'fixed'}, {'OS-EXT-IPS-MAC:mac_addr': 'fa:16:3e:df:b0:8d', 'version': 4, 'addr': '10.1.0.9', 'OS-EXT-IPS:type': 'fixed'}, {'OS-EXT-IPS-MAC:mac_addr': 'fa:16:3e:df:b0:8d', 'version': 4, 'addr': '172.24.5.5', 'OS-EXT-IPS:type': 'floating'}]}
        else:
            addresses = {}
    if image is None:
        image = {'id': '217f3ab1-03e0-4450-bf27-63d52b421e9e', 'links': []}
    if flavor is None:
        flavor = {'id': '64', 'links': []}
    server = {'OS-EXT-STS:task_state': None, 'addresses': addresses, 'links': [], 'image': image, 'OS-EXT-STS:vm_state': 'active', 'OS-SRV-USG:launched_at': '2017-03-23T23:57:38.000000', 'flavor': flavor, 'id': server_id, 'security_groups': [{'name': 'default'}], 'user_id': '9c119f4beaaa438792ce89387362b3ad', 'OS-DCF:diskConfig': 'MANUAL', 'accessIPv4': '', 'accessIPv6': '', 'progress': 0, 'OS-EXT-STS:power_state': 1, 'OS-EXT-AZ:availability_zone': 'nova', 'metadata': {}, 'status': status, 'updated': '2017-03-23T23:57:39Z', 'hostId': '89d165f04384e3ffa4b6536669eb49104d30d6ca832bba2684605dbc', 'OS-SRV-USG:terminated_at': None, 'key_name': None, 'name': name, 'created': '2017-03-23T23:57:12Z', 'tenant_id': PROJECT_ID, 'os-extended-volumes:volumes_attached': [], 'config_drive': 'True'}
    if admin_pass:
        server['adminPass'] = admin_pass
    return json.loads(json.dumps(server))