from google.auth.transport.requests import AuthorizedSession  # type: ignore
import json  # type: ignore
import grpc  # type: ignore
from google.auth.transport.grpc import SslCredentials  # type: ignore
from google.auth import credentials as ga_credentials  # type: ignore
from google.api_core import exceptions as core_exceptions
from google.api_core import retry as retries
from google.api_core import rest_helpers
from google.api_core import rest_streaming
from google.api_core import path_template
from google.api_core import gapic_v1
from cloudsdk.google.protobuf import json_format
from google.iam.v1 import iam_policy_pb2  # type: ignore
from google.iam.v1 import policy_pb2  # type: ignore
from requests import __version__ as requests_version
import dataclasses
import re
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union
import warnings
from google.iam.v1 import iam_policy_pb2  # type: ignore
from google.iam.v1 import policy_pb2  # type: ignore
from cloudsdk.google.protobuf import empty_pb2  # type: ignore
from google.pubsub_v1.types import schema
from google.pubsub_v1.types import schema as gp_schema
from .base import (
def pre_rollback_schema(self, request: schema.RollbackSchemaRequest, metadata: Sequence[Tuple[str, str]]) -> Tuple[schema.RollbackSchemaRequest, Sequence[Tuple[str, str]]]:
    """Pre-rpc interceptor for rollback_schema

        Override in a subclass to manipulate the request or metadata
        before they are sent to the SchemaService server.
        """
    return (request, metadata)