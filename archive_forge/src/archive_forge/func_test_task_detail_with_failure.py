import contextlib
from oslo_utils import uuidutils
from taskflow import exceptions as exc
from taskflow.persistence import models
from taskflow import states
from taskflow.types import failure
def test_task_detail_with_failure(self):
    lb_id = uuidutils.generate_uuid()
    lb_name = 'lb-%s' % lb_id
    lb = models.LogBook(name=lb_name, uuid=lb_id)
    fd = models.FlowDetail('test', uuid=uuidutils.generate_uuid())
    lb.add(fd)
    td = models.TaskDetail('detail-1', uuid=uuidutils.generate_uuid())
    try:
        raise RuntimeError('Woot!')
    except Exception:
        td.failure = failure.Failure()
    fd.add(td)
    with contextlib.closing(self._get_connection()) as conn:
        conn.save_logbook(lb)
        conn.update_flow_details(fd)
        conn.update_atom_details(td)
    with contextlib.closing(self._get_connection()) as conn:
        lb2 = conn.get_logbook(lb_id)
    fd2 = lb2.find(fd.uuid)
    td2 = fd2.find(td.uuid)
    self.assertEqual('Woot!', td2.failure.exception_str)
    self.assertIs(td2.failure.check(RuntimeError), RuntimeError)
    self.assertEqual(td.failure.traceback_str, td2.failure.traceback_str)
    self.assertIsInstance(td2, models.TaskDetail)