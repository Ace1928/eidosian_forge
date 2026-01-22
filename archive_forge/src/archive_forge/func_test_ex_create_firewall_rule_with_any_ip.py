import sys
from types import GeneratorType
from libcloud.test import MockHttp, unittest
from libcloud.utils.py3 import ET, httplib
from libcloud.utils.xml import findall, findtext, fixxpath
from libcloud.common.types import InvalidCredsError
from libcloud.compute.base import Node, NodeLocation, NodeAuthPassword
from libcloud.test.compute import TestCaseMixin
from libcloud.test.secrets import DIMENSIONDATA_PARAMS
from libcloud.test.file_fixtures import ComputeFileFixtures
from libcloud.common.dimensiondata import (
from libcloud.compute.drivers.dimensiondata import DimensionDataNic
from libcloud.compute.drivers.dimensiondata import DimensionDataNodeDriver as DimensionData
def test_ex_create_firewall_rule_with_any_ip(self):
    net = self.driver.ex_get_network_domain('8cdfd607-f429-4df6-9352-162cfc0891be')
    rules = self.driver.ex_list_firewall_rules(net)
    specific_source_ip_rule = list(filter(lambda x: x.name == 'SpecificSourceIP', rules))[0]
    specific_source_ip_rule.source.any_ip = True
    rule = self.driver.ex_create_firewall_rule(net, specific_source_ip_rule, 'FIRST')
    self.assertEqual(rule.id, 'd0a20f59-77b9-4f28-a63b-e58496b73a6c')