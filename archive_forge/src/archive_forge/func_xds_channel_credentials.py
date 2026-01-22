import abc
import contextlib
import enum
import logging
import sys
from grpc import _compression
from grpc._cython import cygrpc as _cygrpc
from grpc._runtime_protos import protos
from grpc._runtime_protos import protos_and_services
from grpc._runtime_protos import services
def xds_channel_credentials(fallback_credentials=None):
    """Creates a ChannelCredentials for use with xDS. This is an EXPERIMENTAL
      API.

    Args:
      fallback_credentials: Credentials to use in case it is not possible to
        establish a secure connection via xDS. If no fallback_credentials
        argument is supplied, a default SSLChannelCredentials is used.
    """
    fallback_credentials = ssl_channel_credentials() if fallback_credentials is None else fallback_credentials
    return ChannelCredentials(_cygrpc.XDSChannelCredentials(fallback_credentials._credentials))