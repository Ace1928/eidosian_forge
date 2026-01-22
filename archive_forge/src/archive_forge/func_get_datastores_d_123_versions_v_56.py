from troveclient import client as base_client
from troveclient.tests import utils
from troveclient.v1 import client
from urllib import parse
def get_datastores_d_123_versions_v_56(self, **kw):
    r = {'version': self.get_datastores_d_123_versions()[2]['versions'][0]}
    return (200, {}, r)