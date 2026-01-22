import uuid
import os_traits
from neutron_lib._i18n import _
from neutron_lib import constants as const
from neutron_lib.placement import constants as place_const
def device_resource_provider_uuid(namespace, host, device, separator=':'):
    """Generate a stable UUID for a physical network device.

    :param namespace: A UUID object identifying a mechanism driver (including
                      its agent).
    :param host: The hostname of the agent.
    :param device: A host-unique name of the physical network device.
    :param separator: A string used in assembling a name for uuid5(). Choose
                      one that cannot occur either in 'host' or 'device'.
                      Optional.
    :returns: A unique and stable UUID identifying a physical network device.
    """
    name = separator.join([host, device])
    return six_uuid5(namespace=namespace, name=name)