import abc
import datetime
from typing import Dict, List, Optional, Sequence, Set, TYPE_CHECKING, Union
import cirq
from cirq_google.cloud import quantum
@abc.abstractmethod
def remove_labels(self, keys: List[str]) -> 'AbstractProgram':
    """Removes labels with given keys from the labels of a previously
        created quantum program.

        Params:
            label_keys: Label keys to remove from the existing program labels.

        Returns:
             This AbstractProgram.
        """