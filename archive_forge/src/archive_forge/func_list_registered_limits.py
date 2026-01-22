import abc
import keystone.conf
from keystone import exception
@abc.abstractmethod
def list_registered_limits(self, hints):
    """List all registered limits.

        :param hints: contains the list of filters yet to be satisfied.
                      Any filters satisfied here will be removed so that
                      the caller will know if any filters remain.

        :returns: a list of dictionaries or an empty registered limit.

        """
    raise exception.NotImplemented()