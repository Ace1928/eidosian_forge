from datetime import datetime
from cinderclient.tests.unit import fakes
from cinderclient.tests.unit.v3 import fakes_base
from cinderclient.v3 import client
def post_attachments(self, **kw):
    if kw['body']['attachment'].get('instance_uuid'):
        return (200, {}, fake_attachment)
    return (200, {}, fake_attachment_without_instance_id)