import abc
from neutron_lib.api.definitions import portbindings
@staticmethod
def provider_network_attribute_updates_supported():
    """Returns the provider network attributes that can be updated

        Possible values: neutron_lib.api.definitions.provider_net.ATTRIBUTES

        :returns: (list) provider network attributes that can be updated in a
                         live network using this driver.
        """
    return []