import abc
from keystone import exception
@abc.abstractmethod
def delete_application_credential(self, application_credential_id):
    """Delete a single application credential.

        :param str application_credential_id: ID of the application credential
            to delete.

        """
    raise exception.NotImplemented()