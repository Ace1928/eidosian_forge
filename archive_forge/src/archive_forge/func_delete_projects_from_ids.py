import abc
import keystone.conf
from keystone import exception
@abc.abstractmethod
def delete_projects_from_ids(self, project_ids):
    """Delete a given list of projects.

        Deletes a list of projects. Ensures no project on the list exists
        after it is successfully called. If an empty list is provided,
        the it is silently ignored. In addition, if a project ID in the list
        of project_ids is not found in the backend, no exception is raised,
        but a message is logged.
        """
    raise exception.NotImplemented()