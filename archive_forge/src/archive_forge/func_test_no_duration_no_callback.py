from oslo_messaging._drivers import common
from oslo_messaging import _utils as utils
from oslo_messaging.tests import utils as test_utils
from unittest import mock
def test_no_duration_no_callback(self):
    t = common.DecayingTimer()
    t.start()
    remaining = t.check_return()
    self.assertIsNone(remaining)