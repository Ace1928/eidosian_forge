from openstack.cloud import _utils
from openstack.cloud import exc
from openstack import exceptions
from openstack.network.v2._proxy import Proxy
def list_firewall_policies(self, filters=None):
    """
        Lists firewall policies.

        :param filters: A dictionary of meta data to use for further filtering.
            Elements of this dictionary may, themselves, be dictionaries.
            Example::

                {
                    'last_name': 'Smith',
                    'other': {
                        'gender': 'Female'
                    }
                }

            OR
            A string containing a jmespath expression for further filtering.
            Example:: "[?last_name==`Smith`] | [?other.gender]==`Female`]"

        :returns: A list of network ``FirewallPolicy`` objects.
        :rtype: list[FirewallPolicy]
        """
    if not filters:
        filters = {}
    return list(self.network.firewall_policies(**filters))