import abc
from oslo_log import log
from keystone import exception
@abc.abstractmethod
def delete_credentials_for_project(self, project_id):
    """Delete all credentials for a project."""
    self._delete_credentials(lambda cr: cr['project_id'] == project_id)