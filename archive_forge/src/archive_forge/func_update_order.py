from openstack.key_manager.v1 import container as _container
from openstack.key_manager.v1 import order as _order
from openstack.key_manager.v1 import secret as _secret
from openstack import proxy
def update_order(self, order, **attrs):
    """Update a order

        :param order: Either the id of a order or a
            :class:`~openstack.key_manager.v1.order.Order` instance.
        :param attrs: The attributes to update on the order represented
            by ``order``.

        :returns: The updated order
        :rtype: :class:`~openstack.key_manager.v1.order.Order`
        """
    return self._update(_order.Order, order, **attrs)