import testtools
from testtools.matchers import HasLength
from ironicclient.tests.unit import utils
from ironicclient.v1 import conductor
def test_conductor_list_detail(self):
    conductors = self.mgr.list(detail=True)
    expect = [('GET', '/v1/conductors/?detail=True', {}, None)]
    self.assertEqual(expect, self.api.calls)
    self.assertEqual(2, len(conductors))