from novaclient import api_versions
from novaclient import base
@api_versions.wraps('2.24')
def live_migration_abort(self, server, migration):
    """
        Cancel an ongoing live migration

        :param server: The :class:`Server` (or its ID)
        :param migration: Migration id that will be cancelled
        :returns: An instance of novaclient.base.TupleWithMeta
        """
    return self._delete('/servers/%s/migrations/%s' % (base.getid(server), base.getid(migration)))