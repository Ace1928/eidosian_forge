import abc
import keystone.conf
from keystone import exception
@abc.abstractmethod
def list_roles_from_ids(self, role_ids):
    """List roles for the provided list of ids.

        :param role_ids: list of ids

        :returns: a list of role_refs.

        This method is used internally by the assignment manager to bulk read
        a set of roles given their ids.

        """
    raise exception.NotImplemented()