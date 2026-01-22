from novaclient import api_versions
from novaclient import base

        Get a list of keypairs.

        :param user_id: Id of key-pairs owner (Admin only).
        :param marker: Begin returning keypairs that appear later in the
                       keypair list than that represented by this keypair name
                       (optional).
        :param limit: maximum number of keypairs to return (optional).
                      Note the API server has a configurable default limit.
                      If no limit is specified here or limit is larger than
                      default, the default limit will be used.
        