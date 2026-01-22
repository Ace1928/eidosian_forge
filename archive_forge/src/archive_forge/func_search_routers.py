from openstack.cloud import _utils
from openstack.cloud import exc
from openstack import exceptions
from openstack.network.v2._proxy import Proxy
def search_routers(self, name_or_id=None, filters=None):
    """Search routers

        :param name_or_id: Name or ID of the desired router.
        :param filters: A dict containing additional filters to use. e.g.
            {'admin_state_up': True}

        :returns: A list of network ``Router`` objects matching the search
            criteria.
        :raises: :class:`~openstack.exceptions.SDKException` if something goes
            wrong during the OpenStack API call.
        """
    query = {}
    if name_or_id:
        query['name'] = name_or_id
    if filters:
        query.update(filters)
    return list(self.network.routers(**query))