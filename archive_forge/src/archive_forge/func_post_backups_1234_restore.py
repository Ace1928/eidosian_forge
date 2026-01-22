from datetime import datetime
from urllib import parse as urlparse
from cinderclient import client as base_client
from cinderclient.tests.unit import fakes
import cinderclient.tests.unit.utils as utils
def post_backups_1234_restore(self, **kw):
    return (200, {}, {'restore': _stub_restore()})