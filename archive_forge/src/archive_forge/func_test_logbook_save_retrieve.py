import contextlib
from oslo_utils import uuidutils
from taskflow import exceptions as exc
from taskflow.persistence import models
from taskflow import states
from taskflow.types import failure
def test_logbook_save_retrieve(self):
    lb_id = uuidutils.generate_uuid()
    lb_meta = {'1': 2}
    lb_name = 'lb-%s' % lb_id
    lb = models.LogBook(name=lb_name, uuid=lb_id)
    lb.meta = lb_meta
    with contextlib.closing(self._get_connection()) as conn:
        self.assertRaises(exc.NotFound, conn.get_logbook, lb_id)
        conn.save_logbook(lb)
    with contextlib.closing(self._get_connection()) as conn:
        lb = conn.get_logbook(lb_id)
    self.assertEqual(lb_name, lb.name)
    self.assertEqual(0, len(lb))
    self.assertEqual(lb_meta, lb.meta)
    self.assertIsNone(lb.updated_at)
    self.assertIsNotNone(lb.created_at)