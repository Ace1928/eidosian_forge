from openstack.key_manager.v1 import container as _container
from openstack.key_manager.v1 import order as _order
from openstack.key_manager.v1 import secret as _secret
from openstack import proxy
def create_order(self, **attrs):
    """Create a new order from attributes

        :param dict attrs: Keyword arguments which will be used to create
            a :class:`~openstack.key_manager.v1.order.Order`,
            comprised of the properties on the Order class.

        :returns: The results of order creation
        :rtype: :class:`~openstack.key_manager.v1.order.Order`
        """
    return self._create(_order.Order, **attrs)