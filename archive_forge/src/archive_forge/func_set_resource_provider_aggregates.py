from openstack.placement.v1 import resource_class as _resource_class
from openstack.placement.v1 import resource_provider as _resource_provider
from openstack.placement.v1 import (
from openstack.placement.v1 import trait as _trait
from openstack import proxy
from openstack import resource
def set_resource_provider_aggregates(self, resource_provider, *aggregates):
    """Update aggregates for a resource provider.

        :param resource_provider: The value can be either the ID of a resource
            provider or an
            :class:`~openstack.placement.v1.resource_provider.ResourceProvider`,
            instance.
        :param aggregates: A list of aggregates. These aggregates will replace
            all aggregates currently present.

        :returns: An instance of
            :class:`~openstack.placement.v1.resource_provider.ResourceProvider`
            with the ``aggregates`` attribute populated with the updated value.
        :raises: :class:`~openstack.exceptions.ResourceNotFound` when no
            resource provider matching the criteria could be found.
        """
    res = self._get_resource(_resource_provider.ResourceProvider, resource_provider)
    return res.set_aggregates(self, aggregates=aggregates)