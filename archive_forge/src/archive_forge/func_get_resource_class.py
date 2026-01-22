from openstack.placement.v1 import resource_class as _resource_class
from openstack.placement.v1 import resource_provider as _resource_provider
from openstack.placement.v1 import (
from openstack.placement.v1 import trait as _trait
from openstack import proxy
from openstack import resource
def get_resource_class(self, resource_class):
    """Get a single resource_class.

        :param resource_class: The value can be either the ID of a resource
            class or an
            :class:`~openstack.placement.v1.resource_class.ResourceClass`,
            instance.

        :returns: An instance of
            :class:`~openstack.placement.v1.resource_class.ResourceClass`
        :raises: :class:`~openstack.exceptions.ResourceNotFound` when no
            resource class matching the criteria could be found.
        """
    return self._get(_resource_class.ResourceClass, resource_class)