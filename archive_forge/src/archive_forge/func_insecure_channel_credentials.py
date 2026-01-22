import copy
import functools
import sys
import warnings
import grpc
from grpc._cython import cygrpc as _cygrpc
def insecure_channel_credentials():
    """Creates a ChannelCredentials for use with an insecure channel.

    THIS IS AN EXPERIMENTAL API.
    """
    return _insecure_channel_credentials