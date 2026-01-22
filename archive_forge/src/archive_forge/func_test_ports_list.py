import copy
import testtools
from testtools.matchers import HasLength
from ironicclient import exc
from ironicclient.tests.unit import utils
import ironicclient.v1.port
def test_ports_list(self):
    ports = self.mgr.list()
    expect = [('GET', '/v1/ports', {}, None)]
    self.assertEqual(expect, self.api.calls)
    self.assertEqual(1, len(ports))