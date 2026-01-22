import abc
from keystone.common import provider_api
import keystone.conf
from keystone import exception
@abc.abstractmethod
def list_projects_associated_with_endpoint_group(self, endpoint_group_id):
    """List all projects associated with endpoint group.

        :param endpoint_group_id: identity of endpoint to associate
        :type endpoint_group_id: string
        :returns: None.

        """
    raise exception.NotImplemented()