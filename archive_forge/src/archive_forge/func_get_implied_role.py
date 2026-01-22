import abc
import keystone.conf
from keystone import exception
@abc.abstractmethod
def get_implied_role(self, prior_role_id, implied_role_id):
    """Get a role inference rule.

        :raises keystone.exception.ImpliedRoleNotFound: If the implied role
            doesn't exist.

        """
    raise exception.NotImplemented()