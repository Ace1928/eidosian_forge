import uuid
from heat.common import short_id
from heat.tests import common
def test_generate_ids(self):
    allowed_chars = [ord(c) for c in u'abcdefghijklmnopqrstuvwxyz234567']
    ids = [short_id.generate_id() for i in range(25)]
    for id in ids:
        self.assertEqual(12, len(id))
        self.assertFalse(id.translate({c: None for c in allowed_chars}))
        self.assertEqual(1, ids.count(id))