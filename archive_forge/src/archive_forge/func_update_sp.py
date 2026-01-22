import abc
from keystone import exception
@abc.abstractmethod
def update_sp(self, sp_id, sp):
    """Update a service provider.

        :param sp_id: id of the service provider
        :type sp_id: string
        :param sp: service prvider object
        :type sp: dict

        :returns: service provider ref
        :rtype: dict

        :raises keystone.exception.ServiceProviderNotFound: If the service
            provider doesn't exist.

        """
    raise exception.NotImplemented()