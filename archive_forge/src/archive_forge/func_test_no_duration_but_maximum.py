from oslo_messaging._drivers import common
from oslo_messaging import _utils as utils
from oslo_messaging.tests import utils as test_utils
from unittest import mock
def test_no_duration_but_maximum(self):
    t = common.DecayingTimer()
    t.start()
    remaining = t.check_return(maximum=2)
    self.assertEqual(2, remaining)