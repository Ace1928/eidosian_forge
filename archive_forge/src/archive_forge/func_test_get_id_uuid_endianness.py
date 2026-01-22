import uuid
from heat.common import short_id
from heat.tests import common
def test_get_id_uuid_endianness(self):
    source = uuid.UUID('ffffffff-00ff-4000-aaaa-aaaaaaaaaaaa')
    self.assertEqual('aaaa77777777', short_id.get_id(source))