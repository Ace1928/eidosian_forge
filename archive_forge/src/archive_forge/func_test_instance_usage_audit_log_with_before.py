from novaclient import api_versions
from novaclient import exceptions
from novaclient.tests.unit import utils
from novaclient.tests.unit.v2 import fakes
def test_instance_usage_audit_log_with_before(self):
    audit_log = self.cs.instance_usage_audit_log.get(before='2016-12-10 13:59:59.999999')
    self.assert_request_id(audit_log, fakes.FAKE_REQUEST_ID_LIST)
    self.cs.assert_called('GET', '/os-instance_usage_audit_log/2016-12-10%2013%3A59%3A59.999999')