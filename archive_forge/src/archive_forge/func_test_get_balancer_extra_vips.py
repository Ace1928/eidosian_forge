import sys
import datetime
from libcloud.test import MockHttp, unittest
from libcloud.utils.py3 import httplib, urlencode
from libcloud.common.types import LibcloudError
from libcloud.loadbalancer.base import Member, Algorithm, LoadBalancer
from libcloud.loadbalancer.types import MemberCondition
from libcloud.test.file_fixtures import OpenStackFixtures, LoadBalancerFileFixtures
from libcloud.loadbalancer.drivers.rackspace import (
def test_get_balancer_extra_vips(self):
    balancer = self.driver.get_balancer(balancer_id='18940')
    self.assertEqual(balancer.extra['virtualIps'], [{'address': '50.56.49.149', 'id': 2359, 'type': 'PUBLIC', 'ipVersion': 'IPV4'}])