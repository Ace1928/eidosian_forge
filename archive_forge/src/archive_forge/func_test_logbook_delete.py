import contextlib
from oslo_utils import uuidutils
from taskflow import exceptions as exc
from taskflow.persistence import models
from taskflow import states
from taskflow.types import failure
def test_logbook_delete(self):
    lb_id = uuidutils.generate_uuid()
    lb_name = 'lb-%s' % lb_id
    lb = models.LogBook(name=lb_name, uuid=lb_id)
    with contextlib.closing(self._get_connection()) as conn:
        self.assertRaises(exc.NotFound, conn.destroy_logbook, lb_id)
    with contextlib.closing(self._get_connection()) as conn:
        conn.save_logbook(lb)
    with contextlib.closing(self._get_connection()) as conn:
        lb2 = conn.get_logbook(lb_id)
        self.assertIsNotNone(lb2)
    with contextlib.closing(self._get_connection()) as conn:
        conn.destroy_logbook(lb_id)
        self.assertRaises(exc.NotFound, conn.destroy_logbook, lb_id)