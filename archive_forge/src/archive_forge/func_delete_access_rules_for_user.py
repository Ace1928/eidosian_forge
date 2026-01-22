import abc
from keystone import exception
@abc.abstractmethod
def delete_access_rules_for_user(self, user_id):
    """Delete all access rules for user.

        This is called when the user itself is deleted.

        :param str user_id: User ID
        """
    raise exception.NotImplemented()