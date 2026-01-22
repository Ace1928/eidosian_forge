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
def composite_channel_credentials(channel_credentials, *call_credentials):
    """Compose a ChannelCredentials and one or more CallCredentials objects.

    Args:
      channel_credentials: A ChannelCredentials object.
      *call_credentials: One or more CallCredentials objects.

    Returns:
      A ChannelCredentials composed of the given ChannelCredentials and
        CallCredentials objects.
    """
    return ChannelCredentials(_cygrpc.CompositeChannelCredentials(tuple((single_call_credentials._credentials for single_call_credentials in call_credentials)), channel_credentials._credentials))