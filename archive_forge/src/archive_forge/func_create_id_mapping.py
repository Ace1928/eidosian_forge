import abc
from keystone.common import provider_api
from keystone import exception
@abc.abstractmethod
def create_id_mapping(self, local_entity, public_id=None):
    """Create and store a mapping to a public_id.

        :param dict local_entity: Containing the entity domain, local ID and
                                  type ('user' or 'group').
        :param public_id: If specified, this will be the public ID.  If this
                          is not specified, a public ID will be generated.
        :returns: public ID

        """
    raise exception.NotImplemented()