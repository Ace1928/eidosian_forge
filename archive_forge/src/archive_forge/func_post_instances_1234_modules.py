from troveclient import client as base_client
from troveclient.tests import utils
from troveclient.v1 import client
from urllib import parse
def post_instances_1234_modules(self, **kw):
    r = {'modules': [self.get_modules()[2]['modules'][0]]}
    return (200, {}, r)