import contextlib
from oslo_utils import uuidutils
from taskflow import exceptions as exc
from taskflow.persistence import models
from taskflow import states
from taskflow.types import failure
def test_flow_detail_lazy_fetch(self):
    lb_id = uuidutils.generate_uuid()
    lb_name = 'lb-%s' % lb_id
    lb = models.LogBook(name=lb_name, uuid=lb_id)
    fd = models.FlowDetail('test', uuid=uuidutils.generate_uuid())
    td = models.TaskDetail('detail-1', uuid=uuidutils.generate_uuid())
    td.version = '4.2'
    fd.add(td)
    lb.add(fd)
    with contextlib.closing(self._get_connection()) as conn:
        conn.save_logbook(lb)
    with contextlib.closing(self._get_connection()) as conn:
        fd2 = conn.get_flow_details(fd.uuid, lazy=True)
        self.assertEqual(0, len(fd2))
        self.assertEqual(1, len(fd))