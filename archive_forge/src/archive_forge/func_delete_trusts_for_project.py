import abc
from keystone import exception
@abc.abstractmethod
def delete_trusts_for_project(self, project_id):
    """Delete all trusts for a project.

        :param project_id: ID of a project to filter trusts by.

        """
    raise exception.NotImplemented()