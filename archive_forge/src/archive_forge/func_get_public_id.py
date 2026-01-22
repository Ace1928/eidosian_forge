import abc
from keystone.common import provider_api
from keystone import exception
@abc.abstractmethod
def get_public_id(self, local_entity):
    """Return the public ID for the given local entity.

        :param dict local_entity: Containing the entity domain, local ID and
                                  type ('user' or 'group').
        :returns: public ID, or None if no mapping is found.

        """
    raise exception.NotImplemented()