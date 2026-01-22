from datetime import datetime
from urllib import parse as urlparse
from cinderclient import client as base_client
from cinderclient.tests.unit import fakes
import cinderclient.tests.unit.utils as utils
def get_cgsnapshots_1234(self, **kw):
    return (200, {}, {'cgsnapshot': _stub_cgsnapshot(id='1234')})