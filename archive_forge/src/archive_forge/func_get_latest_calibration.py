import abc
import datetime
from typing import Dict, List, Optional, overload, TYPE_CHECKING, Union
from google.protobuf.timestamp_pb2 import Timestamp
from cirq_google.engine import calibration
from cirq_google.cloud import quantum
from cirq_google.engine.abstract_processor import AbstractProcessor
from cirq_google.engine.abstract_program import AbstractProgram
@abc.abstractmethod
def get_latest_calibration(self, timestamp: int) -> Optional[calibration.Calibration]:
    """Returns the latest calibration with the provided timestamp or earlier."""