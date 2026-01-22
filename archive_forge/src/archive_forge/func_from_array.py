from dataclasses import dataclass
import logging
import os
import sys
from typing import List, Tuple, Optional, TYPE_CHECKING
@classmethod
def from_array(self, a: 'pyarrow.Array') -> 'PicklableArrayPayload':
    """Create a picklable array payload from an Arrow Array.

        This will recursively accumulate data buffer and metadata payloads that are
        ready for pickling; namely, the data buffers underlying zero-copy slice views
        will be properly truncated.
        """
    return _array_to_array_payload(a)