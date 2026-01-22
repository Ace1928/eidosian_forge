import abc
import keystone.conf
from keystone import exception
@abc.abstractmethod
def list_project_ids_from_domain_ids(self, domain_ids):
    """List project ids for the provided list of domain ids.

        :param domain_ids: list of domain ids

        :returns: a list of project ids owned by the specified domain ids.

        This method is used internally by the assignment manager to bulk read
        a set of project ids given a list of domain ids.

        """
    raise exception.NotImplemented()