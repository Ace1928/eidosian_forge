from openstack.cloud import _utils
from openstack.cloud import exc
from openstack import exceptions
from openstack.network.v2._proxy import Proxy
def get_qos_rule_type_details(self, rule_type, filters=None):
    """Get a QoS rule type details by rule type name.

        :param rule_type: Name of the QoS rule type.
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

        :returns: A network ``QoSRuleType`` object if found, else None.
        """
    if not self._has_neutron_extension('qos'):
        raise exc.OpenStackCloudUnavailableExtension('QoS extension is not available on target cloud')
    if not self._has_neutron_extension('qos-rule-type-details'):
        raise exc.OpenStackCloudUnavailableExtension('qos-rule-type-details extension is not available on target cloud')
    return self.network.get_qos_rule_type(rule_type)