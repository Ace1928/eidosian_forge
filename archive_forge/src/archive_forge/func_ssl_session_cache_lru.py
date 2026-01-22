from grpc._cython import cygrpc as _cygrpc
def ssl_session_cache_lru(capacity):
    """Creates an SSLSessionCache with LRU replacement policy

    Args:
      capacity: Size of the cache

    Returns:
      An SSLSessionCache with LRU replacement policy that can be passed as a value for
      the grpc.ssl_session_cache option to a grpc.Channel. SSL session caches are used
      to store session tickets, which clients can present to resume previous TLS sessions
      with a server.
    """
    return SSLSessionCache(_cygrpc.SSLSessionCacheLRU(capacity))