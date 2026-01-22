from openstack.placement.v1 import resource_class as _resource_class
from openstack.placement.v1 import resource_provider as _resource_provider
from openstack.placement.v1 import (
from openstack.placement.v1 import trait as _trait
from openstack import proxy
from openstack import resource
def create_resource_class(self, **attrs):
    """Create a new resource class from attributes.

        :param attrs: Keyword arguments which will be used to create a
            :class:`~openstack.placement.v1.resource_provider.ResourceClass`,
            comprised of the properties on the ResourceClass class.

        :returns: The results of resource class creation
        :rtype: :class:`~openstack.placement.v1.resource_class.ResourceClass`
        """
    return self._create(_resource_class.ResourceClass, **attrs)