from collections import OrderedDict
import os
import re
from typing import Dict, Optional, Iterable, Iterator, Sequence, Tuple, Type, Union
import pkg_resources
from google.api_core import client_options as client_options_lib
from google.api_core import exceptions as core_exceptions
from google.api_core import gapic_v1
from google.api_core import retry as retries
from google.auth import credentials as ga_credentials
from google.auth.transport import mtls
from google.auth.transport.grpc import SslCredentials
from google.auth.exceptions import MutualTLSChannelError
from google.oauth2 import service_account                         # type: ignore
from cirq_google.cloud.quantum_v1alpha1.services.quantum_engine_service import pagers
from cirq_google.cloud.quantum_v1alpha1.types import engine
from cirq_google.cloud.quantum_v1alpha1.types import quantum
from google.protobuf import any_pb2
from google.protobuf import duration_pb2
from google.protobuf import timestamp_pb2
from .transports.base import QuantumEngineServiceTransport, DEFAULT_CLIENT_INFO
from .transports.grpc import QuantumEngineServiceGrpcTransport
from .transports.grpc_asyncio import QuantumEngineServiceGrpcAsyncIOTransport
@staticmethod
def quantum_reservation_path(project_id: str, processor_id: str, reservation_id: str) -> str:
    """Returns a fully-qualified quantum_reservation string."""
    return f'projects/{project_id}/processors/{processor_id}/reservations/{reservation_id}'