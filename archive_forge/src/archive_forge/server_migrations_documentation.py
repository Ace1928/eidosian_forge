from novaclient import api_versions
from novaclient import base

        Cancel an ongoing live migration

        :param server: The :class:`Server` (or its ID)
        :param migration: Migration id that will be cancelled
        :returns: An instance of novaclient.base.TupleWithMeta
        