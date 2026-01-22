import abc
import datetime
from typing import Dict, List, Optional, Sequence, TYPE_CHECKING, Union
import duet
import cirq
from cirq_google.api import v2
from cirq_google.cloud import quantum
from cirq_google.engine import calibration
@abc.abstractmethod
def supported_languages(self) -> List[str]:
    """Returns the list of processor supported program languages."""