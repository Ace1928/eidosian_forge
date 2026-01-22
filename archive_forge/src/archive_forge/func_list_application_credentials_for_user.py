import abc
from keystone import exception
@abc.abstractmethod
def list_application_credentials_for_user(self, user_id, hints):
    """List application credentials for a user.

        :param str user_id: User ID
        :param hints: contains the list of filters yet to be satisfied.
                      Any filters satisfied here will be removed so that
                      the caller will know if any filters remain.
        """
    raise exception.NotImplemented()