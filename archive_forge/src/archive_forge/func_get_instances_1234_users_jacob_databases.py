from troveclient import client as base_client
from troveclient.tests import utils
from troveclient.v1 import client
from urllib import parse
def get_instances_1234_users_jacob_databases(self, **kw):
    r = {'databases': [self.get_instances_1234_databases()[2]['databases'][0], self.get_instances_1234_databases()[2]['databases'][1]]}
    return (200, {}, r)