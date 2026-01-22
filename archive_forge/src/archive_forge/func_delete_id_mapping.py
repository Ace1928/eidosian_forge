import abc
from keystone.common import provider_api
from keystone import exception
@abc.abstractmethod
def delete_id_mapping(self, public_id):
    """Delete an entry for the given public_id.

        :param public_id: The public ID for the mapping to be deleted.

        The method is silent if no mapping is found.

        """
    raise exception.NotImplemented()