from openstack.key_manager.v1 import container as _container
from openstack.key_manager.v1 import order as _order
from openstack.key_manager.v1 import secret as _secret
from openstack import proxy
def get_container(self, container):
    """Get a single container

        :param container: The value can be the ID of a container or a
            :class:`~openstack.key_manager.v1.container.Container`
            instance.

        :returns: One :class:`~openstack.key_manager.v1.container.Container`
        :raises: :class:`~openstack.exceptions.ResourceNotFound`
            when no resource can be found.
        """
    return self._get(_container.Container, container)