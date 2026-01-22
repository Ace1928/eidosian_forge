import abc
from keystone.common import provider_api
from keystone import exception
@abc.abstractmethod
def get_id_mapping(self, public_id):
    """Return the local mapping.

        :param public_id: The public ID for the mapping required.
        :returns dict: Containing the entity domain, local ID and type. If no
                       mapping is found, it returns None.

        """
    raise exception.NotImplemented()