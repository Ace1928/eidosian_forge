from collections import OrderedDict
import os
import re
from typing import (
from google.cloud.pubsublite_v1 import gapic_version as package_version
from google.api_core import client_options as client_options_lib
from google.api_core import exceptions as core_exceptions
from google.api_core import gapic_v1
from google.api_core import retry as retries
from google.auth import credentials as ga_credentials  # type: ignore
from google.auth.transport import mtls  # type: ignore
from google.auth.transport.grpc import SslCredentials  # type: ignore
from google.auth.exceptions import MutualTLSChannelError  # type: ignore
from google.oauth2 import service_account  # type: ignore
from google.cloud.pubsublite_v1.types import common
from google.cloud.pubsublite_v1.types import topic_stats
from google.longrunning import operations_pb2
from cloudsdk.google.protobuf import timestamp_pb2  # type: ignore
from .transports.base import TopicStatsServiceTransport, DEFAULT_CLIENT_INFO
from .transports.grpc import TopicStatsServiceGrpcTransport
from .transports.grpc_asyncio import TopicStatsServiceGrpcAsyncIOTransport
@staticmethod
def parse_topic_path(path: str) -> Dict[str, str]:
    """Parses a topic path into its component segments."""
    m = re.match('^projects/(?P<project>.+?)/locations/(?P<location>.+?)/topics/(?P<topic>.+?)$', path)
    return m.groupdict() if m else {}