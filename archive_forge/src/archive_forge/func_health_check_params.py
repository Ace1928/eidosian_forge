from tests.integration.route53 import Route53TestCase
from boto.compat import six
from boto.route53.healthcheck import HealthCheck
from boto.route53.record import ResourceRecordSets
def health_check_params(self, **kwargs):
    params = {'ip_addr': '54.217.7.118', 'port': 80, 'hc_type': 'HTTP', 'resource_path': '/testing'}
    params.update(kwargs)
    return params