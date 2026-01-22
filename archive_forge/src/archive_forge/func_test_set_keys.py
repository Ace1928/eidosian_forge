from cinderclient import api_versions
from cinderclient.tests.unit import utils
from cinderclient.tests.unit.v3 import fakes
def test_set_keys(self):
    body = {'qos_specs': dict(k1='v1')}
    qos_id = '1B6B6A04-A927-4AEB-810B-B7BAAD49F57C'
    qos = cs.qos_specs.set_keys(qos_id, body)
    cs.assert_called('PUT', '/qos-specs/%s' % qos_id)
    self._assert_request_id(qos)