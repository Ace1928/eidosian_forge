import abc
from keystone.common import provider_api
import keystone.conf
from keystone import exception
@abc.abstractmethod
def delete_endpoint_group(self, endpoint_group_id):
    """Delete an endpoint group.

        :param endpoint_group_id: identity of endpoint group to delete
        :type endpoint_group_id: string
        :raises keystone.exception.NotFound: If the endpoint group was not
            found.
        :returns: None.

        """
    raise exception.NotImplemented()