import uuid
import os_traits
from neutron_lib._i18n import _
from neutron_lib import constants as const
from neutron_lib.placement import constants as place_const
def resource_request_group_uuid(namespace, qos_rules, separator=':'):
    """Generate a stable UUID for a resource request group.

    :param namespace: A UUID object identifying a port.
    :param qos_rules: A list of QoS rules contributing to the group.
    :param separator: A string used in assembling a name for uuid5(). Optional.
    :returns: A unique and stable UUID identifying a resource request group.
    """
    name = separator.join([rule.id for rule in qos_rules])
    return six_uuid5(namespace=namespace, name=name)