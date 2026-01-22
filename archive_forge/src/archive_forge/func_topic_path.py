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
def topic_path(project: str, location: str, topic: str) -> str:
    """Returns a fully-qualified topic string."""
    return 'projects/{project}/locations/{location}/topics/{topic}'.format(project=project, location=location, topic=topic)