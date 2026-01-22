from datetime import datetime
from urllib import parse as urlparse
from cinderclient import client as base_client
from cinderclient.tests.unit import fakes
import cinderclient.tests.unit.utils as utils
def get_consistencygroups(self, **kw):
    return (200, {}, {'consistencygroups': [_stub_consistencygroup(detailed=False, id='1234'), _stub_consistencygroup(detailed=False, id='4567')]})