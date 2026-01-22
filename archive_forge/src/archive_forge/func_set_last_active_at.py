import abc
from keystone import exception
@abc.abstractmethod
def set_last_active_at(self, user_id):
    """Set the last active at date for the user.

        :param user_id: Unique identifier of the user

        """
    raise exception.NotImplemented()