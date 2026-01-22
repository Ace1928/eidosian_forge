from troveclient import client as base_client
from troveclient.tests import utils
from troveclient.v1 import client
from urllib import parse
def get_cluster_instance_modules(self, **kw):
    return (200, {}, {'modules': [{'auto_apply': False, 'contents': None, 'created': '2018-04-17 05:34:02.84', 'datastore': 'mariadb', 'datastore_version': 'all', 'id': 'module-1', 'is_admin': False, 'md5': 'md5-1', 'message': 'Module.V1', 'name': 'mymod1', 'removed': None, 'status': 'OK', 'tenant': '7f1f041fc291455b83a0b3eb98140808', 'type': 'ping', 'updated': '2018-04-17 05:34:02.84', 'visible': True}, {'auto_apply': False, 'contents': None, 'created': '2018-04-17 05:34:02.84', 'datastore': 'mariadb', 'datastore_version': 'all', 'id': 'module-2', 'is_admin': False, 'md5': 'md5-2', 'message': 'Module.V1', 'name': 'mymod2', 'removed': None, 'status': 'OK', 'tenant': '7f1f041fc291455b83a0b3eb98140808', 'type': 'ping', 'updated': '2018-04-17 05:34:02.84', 'visible': True}]})