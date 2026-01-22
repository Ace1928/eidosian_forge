from openstack.cloud import _utils
from openstack.cloud import exc
from openstack import exceptions
from openstack.network.v2._proxy import Proxy
def list_qos_minimum_bandwidth_rules(self, policy_name_or_id, filters=None):
    """List all available QoS minimum bandwidth rules.

        :param string policy_name_or_id: Name or ID of the QoS policy from
            from rules should be listed.
        :param filters: (optional) A dict of filter conditions to push down
        :returns: A list of network ``QoSMinimumBandwidthRule`` objects.
        :raises: ``:class:`~openstack.exceptions.BadRequestException``` if QoS
            policy will not be found.
        """
    if not self._has_neutron_extension('qos'):
        raise exc.OpenStackCloudUnavailableExtension('QoS extension is not available on target cloud')
    policy = self.network.find_qos_policy(policy_name_or_id)
    if not policy:
        raise exceptions.NotFoundException('QoS policy {name_or_id} not Found.'.format(name_or_id=policy_name_or_id))
    if not filters:
        filters = {}
    return list(self.network.qos_minimum_bandwidth_rules(policy, **filters))