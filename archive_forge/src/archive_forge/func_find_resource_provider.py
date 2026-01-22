from openstack.placement.v1 import resource_class as _resource_class
from openstack.placement.v1 import resource_provider as _resource_provider
from openstack.placement.v1 import (
from openstack.placement.v1 import trait as _trait
from openstack import proxy
from openstack import resource
def find_resource_provider(self, name_or_id, ignore_missing=True):
    """Find a single resource_provider.

        :param name_or_id: The name or ID of a resource provider.
        :param bool ignore_missing: When set to ``False``
            :class:`~openstack.exceptions.ResourceNotFound` will be raised when
            the resource does not exist.  When set to ``True``, None will be
            returned when attempting to find a nonexistent resource.

        :returns: An instance of
            :class:`~openstack.placement.v1.resource_provider.ResourceProvider`
        :raises: :class:`~openstack.exceptions.ResourceNotFound` when no
            resource provider matching the criteria could be found.
        """
    return self._find(_resource_provider.ResourceProvider, name_or_id, ignore_missing=ignore_missing)