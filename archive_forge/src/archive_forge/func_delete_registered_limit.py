import abc
import keystone.conf
from keystone import exception
@abc.abstractmethod
def delete_registered_limit(self, registered_limit_id):
    """Delete an existing registered limit.

        :param registered_limit_id: the registered limit id to delete.

        :raises keystone.exception.RegisteredLimitNotFound: If registered limit
            doesn't exist.

        """
    raise exception.NotImplemented()