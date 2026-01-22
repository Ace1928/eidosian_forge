import abc
import keystone.conf
from keystone import exception
@abc.abstractmethod
def list_projects_in_domain(self, domain_id):
    """List projects in the domain.

        :param domain_id: the driver MUST only return projects
                          within this domain.

        :returns: a list of project_refs or an empty list.

        """
    raise exception.NotImplemented()