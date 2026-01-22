import abc
import keystone.conf
from keystone import exception
@abc.abstractmethod
def check_grant_role_id(self, role_id, user_id=None, group_id=None, domain_id=None, project_id=None, inherited_to_projects=False):
    """Check an assignment/grant role id.

        :raises keystone.exception.RoleAssignmentNotFound: If the role
            assignment doesn't exist.
        :returns: None or raises an exception if grant not found

        """
    raise exception.NotImplemented()