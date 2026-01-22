from openstack.placement.v1 import resource_class as _resource_class
from openstack.placement.v1 import resource_provider as _resource_provider
from openstack.placement.v1 import (
from openstack.placement.v1 import trait as _trait
from openstack import proxy
from openstack import resource
def delete_resource_class(self, resource_class, ignore_missing=True):
    """Delete a resource class

        :param resource_class: The value can be either the ID of a resource
            class or an
            :class:`~openstack.placement.v1.resource_class.ResourceClass`,
            instance.
        :param bool ignore_missing: When set to ``False``
            :class:`~openstack.exceptions.ResourceNotFound` will be raised when
            the resource class does not exist. When set to ``True``, no
            exception will be set when attempting to delete a nonexistent
            resource class.

        :returns: ``None``
        """
    self._delete(_resource_class.ResourceClass, resource_class, ignore_missing=ignore_missing)