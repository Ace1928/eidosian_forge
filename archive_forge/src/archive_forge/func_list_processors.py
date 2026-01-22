import abc
import datetime
from typing import Dict, List, Optional, Sequence, Set, Union
import cirq
from cirq_google.cloud import quantum
from cirq_google.engine import abstract_job, abstract_program, abstract_processor
@abc.abstractmethod
def list_processors(self) -> Sequence[abstract_processor.AbstractProcessor]:
    """Returns all processors in this engine visible to the user."""