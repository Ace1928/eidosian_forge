import abc
import keystone.conf
from keystone import exception
@abc.abstractmethod
def update_role(self, role_id, role):
    """Update an existing role.

        :raises keystone.exception.RoleNotFound: If the role doesn't exist.
        :raises keystone.exception.Conflict: If a duplicate role exists.

        """
    raise exception.NotImplemented()