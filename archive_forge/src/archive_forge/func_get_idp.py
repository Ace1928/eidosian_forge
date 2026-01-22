import abc
from keystone import exception
@abc.abstractmethod
def get_idp(self, idp_id):
    """Get an identity provider by ID.

        :param idp_id: ID of IdP object
        :type idp_id: string
        :raises keystone.exception.IdentityProviderNotFound: If the IdP
            doesn't exist.
        :returns: idp ref
        :rtype: dict

        """
    raise exception.NotImplemented()