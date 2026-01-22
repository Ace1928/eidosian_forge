import contextlib
from oslo_utils import uuidutils
from taskflow import exceptions as exc
from taskflow.persistence import models
from taskflow import states
from taskflow.types import failure
def test_flow_detail_save(self):
    lb_id = uuidutils.generate_uuid()
    lb_name = 'lb-%s' % lb_id
    lb = models.LogBook(name=lb_name, uuid=lb_id)
    fd = models.FlowDetail('test', uuid=uuidutils.generate_uuid())
    lb.add(fd)
    with contextlib.closing(self._get_connection()) as conn:
        self.assertRaises(exc.NotFound, conn.get_logbook, lb_id)
        self.assertRaises(exc.NotFound, conn.update_flow_details, fd)
    with contextlib.closing(self._get_connection()) as conn:
        conn.save_logbook(lb)
        conn.update_flow_details(fd)