from datetime import datetime
from cinderclient.tests.unit import fakes
from cinderclient.tests.unit.v3 import fakes_base
from cinderclient.v3 import client
def get_messages_12345(self, **kw):
    message = {'id': '12345', 'event_id': 'VOLUME_000002', 'user_message': 'Fake Message', 'created_at': '2012-08-27T00:00:00.000000', 'guaranteed_until': '2013-11-12T21:00:00.000000'}
    return (200, {}, {'message': message})