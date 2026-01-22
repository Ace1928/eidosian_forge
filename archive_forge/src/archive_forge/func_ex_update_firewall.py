import sys
import time
import datetime
import itertools
from libcloud.pricing import get_pricing
from libcloud.common.base import LazyObject
from libcloud.common.types import LibcloudError
from libcloud.compute.base import (
from libcloud.common.google import (
from libcloud.compute.types import NodeState
from libcloud.utils.iso8601 import parse_date
from libcloud.compute.providers import Provider
def ex_update_firewall(self, firewall):
    """
        Update a firewall with new values.

        To update, change the attributes of the firewall object and pass the
        updated object to the method.

        :param  firewall: A firewall object with updated values.
        :type   firewall: :class:`GCEFirewall`

        :return:  An object representing the new state of the firewall.
        :rtype:   :class:`GCEFirewall`
        """
    firewall_data = {}
    firewall_data['name'] = firewall.name
    firewall_data['allowed'] = firewall.allowed
    firewall_data['denied'] = firewall.denied
    firewall_data['network'] = firewall.network.extra['selfLink']
    if firewall.source_ranges:
        firewall_data['sourceRanges'] = firewall.source_ranges
    if firewall.source_tags:
        firewall_data['sourceTags'] = firewall.source_tags
    if firewall.source_service_accounts:
        firewall_data['sourceServiceAccounts'] = firewall.source_service_accounts
    if firewall.target_tags:
        firewall_data['targetTags'] = firewall.target_tags
    if firewall.target_service_accounts:
        firewall_data['targetServiceAccounts'] = firewall.target_service_accounts
    if firewall.target_ranges:
        firewall_data['destinationRanges'] = firewall.target_ranges
    if firewall.extra['description']:
        firewall_data['description'] = firewall.extra['description']
    request = '/global/firewalls/%s' % firewall.name
    self.connection.async_request(request, method='PUT', data=firewall_data)
    return self.ex_get_firewall(firewall.name)