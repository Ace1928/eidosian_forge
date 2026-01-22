from openstack.cloud import _utils
from openstack.cloud import exc
from openstack import exceptions
from openstack.network.v2._proxy import Proxy
def list_qos_rule_types(self, filters=None):
    """List all available QoS rule types.

        :param filters: (optional) A dict of filter conditions to push down
        :returns: A list of network ``QosRuleType`` objects.
        """
    if not self._has_neutron_extension('qos'):
        raise exc.OpenStackCloudUnavailableExtension('QoS extension is not available on target cloud')
    if not filters:
        filters = {}
    return list(self.network.qos_rule_types(**filters))