import abc
from keystone import exception
@abc.abstractmethod
def delete_application_credentials_for_user_on_project(self, user_id, project_id):
    """Delete all application credentials for a user on a given project.

        :param str user_id: ID of a user to whose application credentials
            should be deleted.
        :param str project_id: ID of a project on which to filter application
            credentials.

        """
    raise exception.NotImplemented()