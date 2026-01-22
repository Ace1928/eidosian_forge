import abc
from keystone import exception
@abc.abstractmethod
def create_application_credential(self, application_credential, roles):
    """Create a new application credential.

        :param dict application_credential: Application Credential data
        :param list roles: A list of roles that apply to the
                           application_credential.
        :returns: a new application credential
        """
    raise exception.NotImplemented()