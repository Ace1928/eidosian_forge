import abc
import keystone.conf
from keystone import exception
@abc.abstractmethod
def delete_limit(self, limit_id):
    """Delete an existing limit.

        :param limit_id: the limit id to delete.

        :raises keystone.exception.LimitNotFound: If limit doesn't
            exist.

        """
    raise exception.NotImplemented()