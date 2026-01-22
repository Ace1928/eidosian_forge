from datetime import datetime
from urllib import parse as urlparse
from cinderclient import client as base_client
from cinderclient.tests.unit import fakes
import cinderclient.tests.unit.utils as utils
def post_os_snapshot_manage(self, **kw):
    snapshot = _stub_snapshot(id='1234', volume_id='volume_id1')
    snapshot.update(kw['body']['snapshot'])
    return (202, {}, {'snapshot': snapshot})