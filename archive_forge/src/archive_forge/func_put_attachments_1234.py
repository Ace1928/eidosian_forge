from datetime import datetime
from cinderclient.tests.unit import fakes
from cinderclient.tests.unit.v3 import fakes_base
from cinderclient.v3 import client
def put_attachments_1234(self, **kw):
    return (200, {}, {'attachment': {'instance': 1234, 'name': 'attachment-1', 'volume_id': 'fake_volume_1', 'status': 'reserved'}})