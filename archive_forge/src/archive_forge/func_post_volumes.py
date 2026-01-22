from datetime import datetime
from urllib import parse as urlparse
from cinderclient import client as base_client
from cinderclient.tests.unit import fakes
import cinderclient.tests.unit.utils as utils
def post_volumes(self, **kw):
    size = kw['body']['volume'].get('size', 1)
    volume = _stub_volume(id='1234', size=size)
    return (202, {}, {'volume': volume})