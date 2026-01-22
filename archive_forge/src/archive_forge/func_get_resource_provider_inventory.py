from openstack.placement.v1 import resource_class as _resource_class
from openstack.placement.v1 import resource_provider as _resource_provider
from openstack.placement.v1 import (
from openstack.placement.v1 import trait as _trait
from openstack import proxy
from openstack import resource
def get_resource_provider_inventory(self, resource_provider_inventory, resource_provider=None):
    """Get a single resource_provider_inventory

        :param resource_provider_inventory: The value can be either the ID of a
            resource provider inventory or an
            :class:`~openstack.placement.v1.resource_provider_inventory.ResourceProviderInventory`,
            instance.
        :param resource_provider: Either the ID of a resource provider or a
            :class:`~openstack.placement.v1.resource_provider.ResourceProvider`
            instance. This value must be specified when
            ``resource_provider_inventory`` is an ID.

        :returns: An instance of
            :class:`~openstack.placement.v1.resource_provider_inventory.ResourceProviderInventory`
        :raises: :class:`~openstack.exceptions.ResourceNotFound` when no
            resource provider inventory matching the criteria could be found.
        """
    resource_provider_id = self._get_uri_attribute(resource_provider_inventory, resource_provider, 'resource_provider_id')
    return self._get(_resource_provider_inventory.ResourceProviderInventory, resource_provider_inventory, resource_provider_id=resource_provider_id)