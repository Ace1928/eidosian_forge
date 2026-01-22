import testtools
from testtools.matchers import HasLength
from ironicclient.tests.unit import utils
from ironicclient.v1 import conductor
def test_conductor_list(self):
    conductors = self.mgr.list()
    expect = [('GET', '/v1/conductors', {}, None)]
    self.assertEqual(expect, self.api.calls)
    self.assertEqual(2, len(conductors))