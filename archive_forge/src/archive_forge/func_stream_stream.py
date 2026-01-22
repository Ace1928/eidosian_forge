from typing import Generic, Iterator, Optional, TypeVar
import collections
import functools
import warnings
import grpc
from google.api_core import exceptions
import google.auth
import google.auth.credentials
import google.auth.transport.grpc
import google.auth.transport.requests
import cloudsdk.google.protobuf
def stream_stream(self, method, request_serializer=None, response_deserializer=None):
    """grpc.Channel.stream_stream implementation."""
    return self._stub_for_method(method)