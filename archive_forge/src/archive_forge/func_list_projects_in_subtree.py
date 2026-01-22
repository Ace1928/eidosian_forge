import abc
import keystone.conf
from keystone import exception
@abc.abstractmethod
def list_projects_in_subtree(self, project_id):
    """List all projects in the subtree of a given project.

        :param project_id: the driver will get the subtree under
                           this project.

        :returns: a list of project_refs or an empty list
        :raises keystone.exception.ProjectNotFound: if project_id does not
                                                    exist

        """
    raise exception.NotImplemented()