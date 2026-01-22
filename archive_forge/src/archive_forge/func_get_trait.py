from openstack.placement.v1 import resource_class as _resource_class
from openstack.placement.v1 import resource_provider as _resource_provider
from openstack.placement.v1 import (
from openstack.placement.v1 import trait as _trait
from openstack import proxy
from openstack import resource
def get_trait(self, trait):
    """Get a single trait

        :param trait: The value can be either the ID of a trait or an
            :class:`~openstack.placement.v1.trait.Trait`, instance.

        :returns: An instance of
            :class:`~openstack.placement.v1.resource_provider_inventory.ResourceProviderInventory`
        :raises: :class:`~openstack.exceptions.ResourceNotFound` when no
            trait matching the criteria could be found.
        """
    return self._get(_trait.Trait, trait)