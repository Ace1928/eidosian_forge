import abc
import keystone.conf
from keystone import exception
@abc.abstractmethod
def update_limit(self, limit_id, limit):
    """Update existing limits.

        :param limit_id: the id of the limit.
        :param limit: a dict containing the limit attributes to update.

        :returns: the updated limit.
        :raises keystone.exception.LimitNotFound: If limit doesn't
            exist.
        :raises keystone.exception.Conflict: If update to a duplicate limit.

        """
    raise exception.NotImplemented()