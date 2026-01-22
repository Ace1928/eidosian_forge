from cinderclient import api_versions
from cinderclient.tests.unit import utils
from cinderclient.tests.unit.v3 import fakes
def test_associate(self):
    qos_id = '1B6B6A04-A927-4AEB-810B-B7BAAD49F57C'
    type_id = '4230B13A-7A37-4E84-B777-EFBA6FCEE4FF'
    qos = cs.qos_specs.associate(qos_id, type_id)
    cs.assert_called('GET', '/qos-specs/%s/associate?vol_type_id=%s' % (qos_id, type_id))
    self._assert_request_id(qos)