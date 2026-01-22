import collections
import pprint
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import torch
import torch.utils.dlpack
from torch import Tensor
from torch._guards import DuplicateInputs, TracingContext
from torch._prims_common import CUDARngStateHelper
from torch.multiprocessing.reductions import StorageWeakRef
from .. import config
from .collect_metadata_analysis import run_functionalized_fw_and_collect_metadata
from .functional_utils import gen_alias_from_base
from .input_output_analysis import (
from .logging_utils import describe_input, format_guard_bug_msg
from .schemas import (
from .subclass_utils import (
from .utils import (
@wraps(wrapped_compiled_fn)
def debugged_compiled_fn(args):
    new_args = add_dupe_args(remove_dupe_args(args))
    seen: Dict[Any, None] = {}
    for i, (x, y) in enumerate(zip(new_args, args)):
        seen[y] = None
        assert x is y, format_guard_bug_msg(aot_config, f'{describe_input(i, aot_config)} would be a duplicate of {describe_input(add_dupe_map[i], aot_config)}')
    '\n        assert len(seen) == unique_args, format_guard_bug_msg(aot_config,\n            f"there would be {unique_args} distinct arguments"\n        )\n        '
    return wrapped_compiled_fn(args)