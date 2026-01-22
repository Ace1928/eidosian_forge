import abc
import keystone.conf
from keystone import exception
@abc.abstractmethod
def is_leaf_project(self, project_id):
    """Check if a project is a leaf in the hierarchy.

        :param project_id: the driver will check if this project
                           is a leaf in the hierarchy.

        :raises keystone.exception.ProjectNotFound: if project_id does not
                                                    exist

        """
    raise exception.NotImplemented()