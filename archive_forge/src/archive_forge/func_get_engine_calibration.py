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
def get_engine_calibration(processor_id: str, project_id: Optional[str]=None) -> Optional['cirq_google.Calibration']:
    """Returns calibration metrics for a given processor.

    This is a short-cut for creating an engine object, getting the
    processor object, and retrieving the current calibration.
    May return None if no calibration metrics exist for the device.
    """
    return get_engine(project_id).get_processor(processor_id).get_current_calibration()