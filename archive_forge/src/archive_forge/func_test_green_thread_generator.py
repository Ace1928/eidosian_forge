import threading
from unittest import mock
import greenlet
from oslo_config import cfg
from oslotest import base
from oslo_reports.generators import conf as os_cgen
from oslo_reports.generators import threading as os_tgen
from oslo_reports.generators import version as os_pgen
from oslo_reports.models import threading as os_tmod
def test_green_thread_generator(self):
    curr_g = greenlet.getcurrent()
    model = os_tgen.GreenThreadReportGenerator()()
    self.assertTrue(len(model.keys()) >= 1)
    was_ok = False
    for tm in model.values():
        if tm.stack_trace == os_tmod.StackTraceModel(curr_g.gr_frame):
            was_ok = True
            break
    self.assertTrue(was_ok)
    model.set_current_view_type('text')
    self.assertIsNotNone(str(model))