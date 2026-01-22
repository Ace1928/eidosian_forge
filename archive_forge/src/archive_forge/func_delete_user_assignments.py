import abc
import keystone.conf
from keystone import exception
@abc.abstractmethod
def delete_user_assignments(self, user_id):
    """Delete all assignments for a user.

        :raises keystone.exception.RoleNotFound: If the role doesn't exist.

        """
    raise exception.NotImplemented()