import sys
import datetime
from libcloud.test import MockHttp, unittest
from libcloud.utils.py3 import httplib, urlencode
from libcloud.common.types import LibcloudError
from libcloud.loadbalancer.base import Member, Algorithm, LoadBalancer
from libcloud.loadbalancer.types import MemberCondition
from libcloud.test.file_fixtures import OpenStackFixtures, LoadBalancerFileFixtures
from libcloud.loadbalancer.drivers.rackspace import (
def test_ex_update_balancer_error_page(self):
    balancer = self.driver.get_balancer(balancer_id='8290')
    content = '<html>Generic Error Page</html>'
    balancer = self.driver.ex_update_balancer_error_page(balancer, content)
    error_page_content = self.driver.ex_get_balancer_error_page(balancer)
    self.assertEqual(content, error_page_content)