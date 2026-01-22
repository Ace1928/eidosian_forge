import abc
import keystone.conf
from keystone import exception
@abc.abstractmethod
def delete_project_assignments(self, project_id):
    """Delete all assignments for a project.

        :raises keystone.exception.ProjectNotFound: If the project doesn't
            exist.

        """
    raise exception.NotImplemented()