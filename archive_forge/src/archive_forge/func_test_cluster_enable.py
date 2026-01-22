import ddt
from cinderclient import api_versions
from cinderclient import exceptions as exc
from cinderclient.tests.unit import utils
from cinderclient.tests.unit.v3 import fakes
def test_cluster_enable(self):
    body = {'binary': 'cinder-volume', 'name': 'cluster@lvmdriver-1'}
    result = cs.clusters.update(body['name'], body['binary'], False, disabled_reason='is ignored')
    self._assert_call('/clusters/enable', False, method='PUT', body=body)
    self._assert_request_id(result)
    self._check_fields_present([result], False)