import abc
import string
from keystone import exception
@abc.abstractmethod
def list_access_tokens(self, user_id):
    """List access tokens.

        :param user_id: search for access tokens authorized by given user id
        :type user_id: string
        :returns: list of access tokens the user has authorized

        """
    raise exception.NotImplemented()