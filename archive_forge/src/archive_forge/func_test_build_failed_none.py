import collections
from unittest import mock
from heatclient import exc
from heatclient.osc.v1 import stack_failures
from heatclient.tests.unit.osc.v1 import fakes as orchestration_fakes
def test_build_failed_none(self):
    self.stack = mock.MagicMock(id='123', status='COMPLETE', stack_name='stack')
    failures = self.cmd._build_failed_resources('stack')
    expected = collections.OrderedDict()
    self.assertEqual(expected, failures)