from datetime import datetime
from urllib import parse as urlparse
from cinderclient import client as base_client
from cinderclient.tests.unit import fakes
import cinderclient.tests.unit.utils as utils
def get_volumes_1234_metadata(self, **kw):
    r = {'metadata': {'k1': 'v1', 'k2': 'v2', 'k3': 'v3'}}
    return (200, {}, r)