from openstack.cloud import _utils
from openstack.cloud import exc
from openstack import exceptions
from openstack.network.v2._proxy import Proxy
def search_qos_dscp_marking_rules(self, policy_name_or_id, rule_id=None, filters=None):
    """Search QoS DSCP marking rules

        :param string policy_name_or_id: Name or ID of the QoS policy to which
            rules should be associated.
        :param string rule_id: ID of searched rule.
        :param filters: a dict containing additional filters to use. e.g.
            {'dscp_mark': 32}

        :returns: A list of network ``QoSDSCPMarkingRule`` objects matching the
            search criteria.
        :raises: :class:`~openstack.exceptions.SDKException` if something goes
            wrong during the OpenStack API call.
        """
    rules = self.list_qos_dscp_marking_rules(policy_name_or_id, filters)
    return _utils._filter_list(rules, rule_id, filters)