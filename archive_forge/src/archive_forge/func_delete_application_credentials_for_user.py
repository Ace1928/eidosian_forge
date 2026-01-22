import abc
from keystone import exception
@abc.abstractmethod
def delete_application_credentials_for_user(self, user_id):
    """Delete all application credentials for a user.

        :param user_id: ID of a user to whose application credentials should
            be deleted.

        """
    raise exception.NotImplemented()