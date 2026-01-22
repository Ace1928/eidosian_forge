import functools
from typing import Any, Dict, List, Optional, TYPE_CHECKING
from cirq import protocols
from cirq.ops import raw_types
def handle_transition_at(k):
    chunk = text[last_transition:k]
    if was_on_digits:
        chunk = chunk.rjust(8, '0') + ':' + str(len(chunk))
    chunks.append(chunk)