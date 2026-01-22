import collections
import functools
import numbers
import sys
from torch.utils.data.datapipes._hook_iterator import hook_iterator, _SnapshotState
from typing import (Any, Dict, Iterator, Generic, List, Set, Tuple, TypeVar, Union,
from typing import _eval_type, _tp_cache, _type_check, _type_repr  # type: ignore[attr-defined]
from typing import ForwardRef
from abc import ABCMeta
from typing import _GenericAlias  # type: ignore[attr-defined, no-redef]
@functools.wraps(reset_func)
def conditional_reset(*args, **kwargs):
    """
                Only execute DataPipe's `reset()` method if `_SnapshotState` is `Iterating` or `NotStarted`.

                This allows recently restored DataPipe to preserve its restored state during the initial `__iter__` call.
                """
    datapipe = args[0]
    if datapipe._snapshot_state in (_SnapshotState.Iterating, _SnapshotState.NotStarted):
        datapipe._number_of_samples_yielded = 0
        datapipe._fast_forward_iterator = None
        reset_func(*args, **kwargs)
    datapipe._snapshot_state = _SnapshotState.Iterating