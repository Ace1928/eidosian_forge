from openstack.placement.v1 import resource_class as _resource_class
from openstack.placement.v1 import resource_provider as _resource_provider
from openstack.placement.v1 import (
from openstack.placement.v1 import trait as _trait
from openstack import proxy
from openstack import resource
def update_resource_provider(self, resource_provider, **attrs):
    """Update a resource provider

        :param resource_provider: The value can be either the ID of a resource
            provider or an
            :class:`~openstack.placement.v1.resource_provider.ResourceProvider`,
            instance.
        :param attrs: The attributes to update on the resource provider
            represented by ``resource_provider``.

        :returns: The updated resource provider
        :rtype: :class:`~openstack.placement.v1.resource_provider.ResourceProvider`
        """
    return self._update(_resource_provider.ResourceProvider, resource_provider, **attrs)