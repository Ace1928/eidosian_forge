from troveclient import client as base_client
from troveclient.tests import utils
from troveclient.v1 import client
from urllib import parse
def get_modules_4321_instances(self, **kw):
    if kw.get('count_only', False):
        return self.get_instance_counts()
    return self.get_instances()