from collections import OrderedDict
import os
import re
from typing import (
import warnings
from google.api_core import client_options as client_options_lib
from google.api_core import exceptions as core_exceptions
from google.api_core import gapic_v1
from google.api_core import retry as retries
from google.auth import credentials as ga_credentials  # type: ignore
from google.auth.exceptions import MutualTLSChannelError  # type: ignore
from google.auth.transport import mtls  # type: ignore
from google.auth.transport.grpc import SslCredentials  # type: ignore
from google.oauth2 import service_account  # type: ignore
from google.cloud.texttospeech_v1 import gapic_version as package_version
from google.api_core import operation  # type: ignore
from google.api_core import operation_async  # type: ignore
from google.longrunning import operations_pb2  # type: ignore
from google.cloud.texttospeech_v1.types import cloud_tts_lrs
from .transports.base import (
from .transports.grpc import TextToSpeechLongAudioSynthesizeGrpcTransport
from .transports.grpc_asyncio import TextToSpeechLongAudioSynthesizeGrpcAsyncIOTransport
from .transports.rest import TextToSpeechLongAudioSynthesizeRestTransport
@staticmethod
def parse_model_path(path: str) -> Dict[str, str]:
    """Parses a model path into its component segments."""
    m = re.match('^projects/(?P<project>.+?)/locations/(?P<location>.+?)/models/(?P<model>.+?)$', path)
    return m.groupdict() if m else {}