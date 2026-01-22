from troveclient import client as base_client
from troveclient.tests import utils
from troveclient.v1 import client
from urllib import parse
def get_instances_1234_users_jacob(self, **kw):
    r = {'user': self.get_instances_1234_users()[2]['users'][0]}
    return (200, {}, r)