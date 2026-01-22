from openstack.key_manager.v1 import container as _container
from openstack.key_manager.v1 import order as _order
from openstack.key_manager.v1 import secret as _secret
from openstack import proxy
def find_container(self, name_or_id, ignore_missing=True):
    """Find a single container

        :param name_or_id: The name or ID of a container.
        :param bool ignore_missing: When set to ``False``
            :class:`~openstack.exceptions.ResourceNotFound` will be
            raised when the resource does not exist.
            When set to ``True``, None will be returned when
            attempting to find a nonexistent resource.
        :returns: One :class:`~openstack.key_manager.v1.container.Container`
            or None
        """
    return self._find(_container.Container, name_or_id, ignore_missing=ignore_missing)