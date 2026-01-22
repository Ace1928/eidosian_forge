import datetime
import enum
import random
import string
from typing import Dict, List, Optional, Sequence, Set, TypeVar, Union, TYPE_CHECKING
import duet
import google.auth
from google.protobuf import any_pb2
import cirq
from cirq._compat import deprecated
from cirq_google.api import v2
from cirq_google.engine import (
from cirq_google.cloud import quantum
from cirq_google.engine.result_type import ResultType
from cirq_google.serialization import CIRCUIT_SERIALIZER, Serializer
from cirq_google.serialization.arg_func_langs import arg_to_proto
def get_engine_device(processor_id: str, project_id: Optional[str]=None) -> cirq.Device:
    """Returns a `Device` object for a given processor.

    This is a short-cut for creating an engine object, getting the
    processor object, and retrieving the device.
    """
    return get_engine(project_id).get_processor(processor_id).get_device()