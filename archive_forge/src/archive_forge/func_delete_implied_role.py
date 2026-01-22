import abc
import keystone.conf
from keystone import exception
@abc.abstractmethod
def delete_implied_role(self, prior_role_id, implied_role_id):
    """Delete a role inference rule.

        :raises keystone.exception.ImpliedRoleNotFound: If the implied role
            doesn't exist.

        """
    raise exception.NotImplemented()