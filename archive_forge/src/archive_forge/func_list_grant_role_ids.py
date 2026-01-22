import abc
import keystone.conf
from keystone import exception
@abc.abstractmethod
def list_grant_role_ids(self, user_id=None, group_id=None, domain_id=None, project_id=None, inherited_to_projects=False):
    """List role ids for assignments/grants."""
    raise exception.NotImplemented()