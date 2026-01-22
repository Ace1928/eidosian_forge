from urllib import parse
from saharaclient.api import base
def update_keypair(self, cluster_id):
    """Reflect an updated keypair on the cluster."""
    data = {'update_keypair': True}
    return self._patch('/clusters/%s' % cluster_id, data)