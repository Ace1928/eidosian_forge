import copy
import testtools
from testtools.matchers import HasLength
from ironicclient import exc
from ironicclient.tests.unit import utils
import ironicclient.v1.port
def test_port_show_fields(self):
    port = self.mgr.get(PORT['uuid'], fields=['uuid', 'address'])
    expect = [('GET', '/v1/ports/%s?fields=uuid,address' % PORT['uuid'], {}, None)]
    self.assertEqual(expect, self.api.calls)
    self.assertEqual(PORT['uuid'], port.uuid)
    self.assertEqual(PORT['address'], port.address)