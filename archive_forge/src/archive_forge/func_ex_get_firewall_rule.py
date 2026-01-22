import re
import sys
from libcloud.utils.py3 import ET, urlencode, basestring, ensure_string
from libcloud.utils.xml import findall, findtext, fixxpath
from libcloud.compute.base import (
from libcloud.common.nttcis import (
from libcloud.compute.types import Provider, NodeState
from libcloud.common.exceptions import BaseHTTPError
def ex_get_firewall_rule(self, network_domain, rule_id):
    locations = self.list_locations()
    rule = self.connection.request_with_orgId_api_2('network/firewallRule/%s' % rule_id).object
    return self._to_firewall_rule(rule, locations, network_domain)