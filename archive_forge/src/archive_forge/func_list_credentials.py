import abc
from oslo_log import log
from keystone import exception
@abc.abstractmethod
def list_credentials(self, hints):
    """List all credentials.

        :param hints: contains the list of filters yet to be satisfied.
                      Any filters satisfied here will be removed so that
                      the caller will know if any filters remain.

        :returns: a list of credential_refs or an empty list.

        """
    raise exception.NotImplemented()