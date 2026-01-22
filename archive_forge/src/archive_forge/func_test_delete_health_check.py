from tests.integration.route53 import Route53TestCase
from boto.compat import six
from boto.route53.healthcheck import HealthCheck
from boto.route53.record import ResourceRecordSets
def test_delete_health_check(self):
    hc = HealthCheck(ip_addr='54.217.7.118', port=80, hc_type='HTTP', resource_path='/testing')
    result = self.conn.create_health_check(hc)
    hc_id = result['CreateHealthCheckResponse']['HealthCheck']['Id']
    result = self.conn.get_list_health_checks()
    found = False
    for hc in result['ListHealthChecksResponse']['HealthChecks']:
        if hc['Id'] == hc_id:
            found = True
            break
    self.assertTrue(found)
    result = self.conn.delete_health_check(hc_id)
    result = self.conn.get_list_health_checks()
    for hc in result['ListHealthChecksResponse']['HealthChecks']:
        self.assertFalse(hc['Id'] == hc_id)