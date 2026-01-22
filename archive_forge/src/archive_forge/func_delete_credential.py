import abc
from oslo_log import log
from keystone import exception
@abc.abstractmethod
def delete_credential(self, credential_id):
    """Delete an existing credential.

        :raises keystone.exception.CredentialNotFound: If credential doesn't
            exist.

        """
    raise exception.NotImplemented()