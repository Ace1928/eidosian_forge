from troveclient import base
from troveclient import common
from troveclient.v1 import users
def is_cluster_root_enabled(self, cluster):
    """Returns whether root is enabled for the cluster."""
    return self._is_root_enabled(self.clusters_url % base.getid(cluster))