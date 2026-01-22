import abc
from keystone import exception
@abc.abstractmethod
def update_idp(self, idp_id, idp):
    """Update an identity provider by ID.

        :param idp_id: ID of IdP object
        :type idp_id: string
        :param idp: idp object
        :type idp: dict
        :raises keystone.exception.IdentityProviderNotFound: If the IdP
            doesn't exist.
        :returns: idp ref
        :rtype: dict

        """
    raise exception.NotImplemented()