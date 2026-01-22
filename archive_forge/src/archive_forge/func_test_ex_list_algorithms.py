import sys
import datetime
from libcloud.test import MockHttp, unittest
from libcloud.utils.py3 import httplib, urlencode
from libcloud.common.types import LibcloudError
from libcloud.loadbalancer.base import Member, Algorithm, LoadBalancer
from libcloud.loadbalancer.types import MemberCondition
from libcloud.test.file_fixtures import OpenStackFixtures, LoadBalancerFileFixtures
from libcloud.loadbalancer.drivers.rackspace import (
def test_ex_list_algorithms(self):
    algorithms = self.driver.ex_list_algorithm_names()
    self.assertTrue('RANDOM' in algorithms)
    self.assertTrue('ROUND_ROBIN' in algorithms)
    self.assertTrue('LEAST_CONNECTIONS' in algorithms)
    self.assertTrue('WEIGHTED_ROUND_ROBIN' in algorithms)
    self.assertTrue('WEIGHTED_LEAST_CONNECTIONS' in algorithms)