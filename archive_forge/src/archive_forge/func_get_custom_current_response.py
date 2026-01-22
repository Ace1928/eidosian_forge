from unittest import mock
from keystoneauth1 import identity
from keystoneauth1 import session
def get_custom_current_response(min_version='1.0', max_version='1.1'):
    return {'version': {'id': 'v1', 'status': 'CURRENT', 'min_version': min_version, 'max_version': max_version, 'links': [{'rel': 'self', 'href': 'http://192.168.1.23/key-manager/v1/'}, {'rel': 'describedby', 'type': 'text/html', 'href': 'https://docs.openstack.org/'}]}}