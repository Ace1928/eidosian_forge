import abc
import keystone.conf
from keystone import exception
@abc.abstractmethod
def delete_group_assignments(self, group_id):
    """Delete all assignments for a group.

        :raises keystone.exception.RoleNotFound: If the role doesn't exist.

        """
    raise exception.NotImplemented()