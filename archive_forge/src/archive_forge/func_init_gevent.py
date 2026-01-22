from grpc._cython import cygrpc as _cygrpc
def init_gevent():
    """Patches gRPC's libraries to be compatible with gevent.

    This must be called AFTER the python standard lib has been patched,
    but BEFORE creating and gRPC objects.

    In order for progress to be made, the application must drive the event loop.
    """
    _cygrpc.init_grpc_gevent()