import ddt
from cinderclient import api_versions
from cinderclient import exceptions as exc
from cinderclient.tests.unit import utils
from cinderclient.tests.unit.v3 import fakes
def test_cluster_disable(self):
    body = {'binary': 'cinder-volume', 'name': 'cluster@lvmdriver-1', 'disabled_reason': 'is passed'}
    result = cs.clusters.update(body['name'], body['binary'], True, body['disabled_reason'])
    self._assert_call('/clusters/disable', False, method='PUT', body=body)
    self._assert_request_id(result)
    self._check_fields_present([result], False)