from troveclient import client as base_client
from troveclient.tests import utils
from troveclient.v1 import client
from urllib import parse
def get_security_groups(self, **kw):
    return (200, {}, {'security_groups': [{'instance_id': '1234', 'updated': '2015-05-16T17:29:45', 'name': 'SecGroup_1234', 'created': '2015-05-16T17:29:45', 'rules': [{'to_port': 3306, 'cidr': '0.0.0.0/0', 'from_port': 3306, 'protocol': 'tcp', 'id': '1'}], 'id': '2', 'description': 'Security Group for 1234'}]})