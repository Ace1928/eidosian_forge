import abc
import functools
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, TYPE_CHECKING, Union
from typing_extensions import Self
from cirq import ops, protocols
@staticmethod
def for_qid_shape(qid_shape: Sequence[int], start: int=0, step: int=1) -> List['LineQid']:
    """Returns a range of line qids for each entry in `qid_shape` with
        matching dimension.

        Args:
            qid_shape: A sequence of dimensions for each `LineQid` to create.
            start: The x coordinate of the first `LineQid`.
            step: The amount to increment each x coordinate.
        """
    return [LineQid(start + step * i, dimension=dimension) for i, dimension in enumerate(qid_shape)]