from troveclient import client as base_client
from troveclient.tests import utils
from troveclient.v1 import client
from urllib import parse
def get_volume_types(self, **kw):
    return (200, {}, {'volume_types': [{'id': '1', 'name': 'vt_1', 'description': 'Volume type #1', 'is_public': False}, {'id': '10', 'name': 'volume_type_2', 'description': 'Volume type #2', 'is_public': True}]})