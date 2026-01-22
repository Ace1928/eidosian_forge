from openstack.cloud import _utils
from openstack.cloud import exc
from openstack import exceptions
from openstack.network.v2._proxy import Proxy
def search_subnets(self, name_or_id=None, filters=None):
    """Search subnets

        :param name_or_id: Name or ID of the desired subnet.
        :param filters: A dict containing additional filters to use. e.g.
            {'enable_dhcp': True}

        :returns: A list of network ``Subnet`` objects matching the search
            criteria.
        :raises: :class:`~openstack.exceptions.SDKException` if something goes
            wrong during the OpenStack API call.
        """
    query = {}
    if name_or_id:
        query['name'] = name_or_id
    if filters:
        query.update(filters)
    return list(self.network.subnets(**query))