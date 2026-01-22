from collections import OrderedDict
import os
import re
from typing import Dict, Mapping, MutableMapping, MutableSequence, Optional, Iterable, Iterator, Sequence, Tuple, Type, Union, cast
from googlecloudsdk.generated_clients.gapic_clients.logging_v2 import gapic_version as package_version
from google.api_core import client_options as client_options_lib
from google.api_core import exceptions as core_exceptions
from google.api_core import gapic_v1
from google.api_core import retry as retries
from google.auth import credentials as ga_credentials             # type: ignore
from google.auth.transport import mtls                            # type: ignore
from google.auth.transport.grpc import SslCredentials             # type: ignore
from google.auth.exceptions import MutualTLSChannelError          # type: ignore
from google.oauth2 import service_account                         # type: ignore
from google.api import monitored_resource_pb2  # type: ignore
from googlecloudsdk.generated_clients.gapic_clients.logging_v2.services.logging_service_v2 import pagers
from googlecloudsdk.generated_clients.gapic_clients.logging_v2.types import log_entry
from googlecloudsdk.generated_clients.gapic_clients.logging_v2.types import logging
from .transports.base import LoggingServiceV2Transport, DEFAULT_CLIENT_INFO
from .transports.grpc import LoggingServiceV2GrpcTransport
from .transports.grpc_asyncio import LoggingServiceV2GrpcAsyncIOTransport
from .transports.rest import LoggingServiceV2RestTransport
@staticmethod
def parse_log_path(path: str) -> Dict[str, str]:
    """Parses a log path into its component segments."""
    m = re.match('^projects/(?P<project>.+?)/logs/(?P<log>.+?)$', path)
    return m.groupdict() if m else {}