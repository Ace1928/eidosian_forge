import re
from typing import Callable, Dict, Optional, Sequence, Tuple, Union
from requests import __version__ as requests_version
from google.api_core import exceptions as core_exceptions  # type: ignore
from google.api_core import gapic_v1  # type: ignore
from google.api_core import path_template  # type: ignore
from google.api_core import rest_helpers  # type: ignore
from google.api_core import retry as retries  # type: ignore
from google.auth import credentials as ga_credentials  # type: ignore
from google.auth.transport.requests import AuthorizedSession  # type: ignore
from google.longrunning import operations_pb2  # type: ignore
from cloudsdk.google.protobuf import empty_pb2  # type: ignore
from cloudsdk.google.protobuf import json_format  # type: ignore
import grpc
from .base import DEFAULT_CLIENT_INFO as BASE_DEFAULT_CLIENT_INFO, OperationsTransport
@property
def delete_operation(self) -> Callable[[operations_pb2.DeleteOperationRequest], empty_pb2.Empty]:
    return self._delete_operation