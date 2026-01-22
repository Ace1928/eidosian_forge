from openstack.placement.v1 import resource_class as _resource_class
from openstack.placement.v1 import resource_provider as _resource_provider
from openstack.placement.v1 import (
from openstack.placement.v1 import trait as _trait
from openstack import proxy
from openstack import resource
def resource_provider_inventories(self, resource_provider, **query):
    """Retrieve a generator of resource provider inventories

        :param resource_provider: Either the ID of a resource provider or a
            :class:`~openstack.placement.v1.resource_provider.ResourceProvider`
            instance.
        :param query: Optional query parameters to be sent to limit
            the resources being returned.

        :returns: A generator of resource provider inventory instances.
        """
    resource_provider_id = resource.Resource._get_id(resource_provider)
    return self._list(_resource_provider_inventory.ResourceProviderInventory, resource_provider_id=resource_provider_id, **query)