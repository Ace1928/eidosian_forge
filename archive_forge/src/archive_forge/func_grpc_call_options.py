import abc
import grpc
def grpc_call_options(disable_compression=False, credentials=None):
    """Creates a GRPCCallOptions value to be passed at RPC invocation.

    All parameters are optional and should always be passed by keyword.

    Args:
      disable_compression: A boolean indicating whether or not compression should
        be disabled for the request object of the RPC. Only valid for
        request-unary RPCs.
      credentials: A CallCredentials object to use for the invoked RPC.
    """
    return GRPCCallOptions(disable_compression, None, credentials)