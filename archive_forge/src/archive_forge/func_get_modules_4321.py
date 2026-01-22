from troveclient import client as base_client
from troveclient.tests import utils
from troveclient.v1 import client
from urllib import parse
def get_modules_4321(self, **kw):
    r = {'module': self.get_modules()[2]['modules'][0]}
    return (200, {}, r)