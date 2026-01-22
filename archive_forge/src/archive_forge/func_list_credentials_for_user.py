import abc
from oslo_log import log
from keystone import exception
@abc.abstractmethod
def list_credentials_for_user(self, user_id, type=None):
    """List credentials for a user.

        :param user_id: ID of a user to filter credentials by.
        :param type: type of credentials to filter on.

        :returns: a list of credential_refs or an empty list.

        """
    raise exception.NotImplemented()