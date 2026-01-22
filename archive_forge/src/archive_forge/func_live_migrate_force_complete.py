from novaclient import api_versions
from novaclient import base
@api_versions.wraps('2.22')
def live_migrate_force_complete(self, server, migration):
    """
        Force on-going live migration to complete

        :param server: The :class:`Server` (or its ID)
        :param migration: Migration id that will be forced to complete
        :returns: An instance of novaclient.base.TupleWithMeta
        """
    body = {'force_complete': None}
    resp, body = self.api.client.post('/servers/%s/migrations/%s/action' % (base.getid(server), base.getid(migration)), body=body)
    return self.convert_into_with_meta(body, resp)