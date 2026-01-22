from openstack.placement.v1 import resource_class as _resource_class
from openstack.placement.v1 import resource_provider as _resource_provider
from openstack.placement.v1 import (
from openstack.placement.v1 import trait as _trait
from openstack import proxy
from openstack import resource
def update_resource_provider_inventory(self, resource_provider_inventory, resource_provider=None, *, resource_provider_generation=None, **attrs):
    """Update a resource provider's inventory

        :param resource_provider_inventory: The value can be either the ID of a resource
            provider inventory or an
            :class:`~openstack.placement.v1.resource_provider_inventory.ResourceProviderInventory`,
            instance.
        :param resource_provider: Either the ID of a resource provider or a
            :class:`~openstack.placement.v1.resource_provider.ResourceProvider`
            instance. This value must be specified when
            ``resource_provider_inventory`` is an ID.
        :attrs kwargs: The attributes to update on the resource provider inventory
            represented by ``resource_provider_inventory``.

        :returns: The updated resource provider inventory
        :rtype: :class:`~openstack.placement.v1.resource_provider_inventory.ResourceProviderInventory`
        """
    resource_provider_id = self._get_uri_attribute(resource_provider_inventory, resource_provider, 'resource_provider_id')
    return self._update(_resource_provider_inventory.ResourceProviderInventory, resource_provider_inventory, resource_provider_id=resource_provider_id, resource_provider_generation=resource_provider_generation, **attrs)