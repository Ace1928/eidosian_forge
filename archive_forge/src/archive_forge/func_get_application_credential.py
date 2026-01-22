import abc
from keystone import exception
@abc.abstractmethod
def get_application_credential(self, application_credential_id):
    """Get an application credential by the credential id.

        :param str application_credential_id: Application Credential ID
        """
    raise exception.NotImplemented()