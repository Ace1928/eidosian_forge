import abc
from oslo_log import log
from keystone import exception
@abc.abstractmethod
def delete_credentials_for_user(self, user_id):
    """Delete all credentials for a user."""
    self._delete_credentials(lambda cr: cr['user_id'] == user_id)