from ..push import PushResult
from .errors import GitSmartRemoteNotSupported
def remote_divergence(old_sha, new_sha, store):
    if old_sha is None:
        return False
    if not isinstance(old_sha, bytes):
        raise TypeError(old_sha)
    if not isinstance(new_sha, bytes):
        raise TypeError(new_sha)
    from breezy.graph import Graph
    graph = Graph(ObjectStoreParentsProvider(store))
    return not graph.is_ancestor(old_sha, new_sha)