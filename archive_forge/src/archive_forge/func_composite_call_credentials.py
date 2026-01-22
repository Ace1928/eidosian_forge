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
def composite_call_credentials(*call_credentials):
    """Compose multiple CallCredentials to make a new CallCredentials.

    Args:
      *call_credentials: At least two CallCredentials objects.

    Returns:
      A CallCredentials object composed of the given CallCredentials objects.
    """
    return CallCredentials(_cygrpc.CompositeCallCredentials(tuple((single_call_credentials._credentials for single_call_credentials in call_credentials))))