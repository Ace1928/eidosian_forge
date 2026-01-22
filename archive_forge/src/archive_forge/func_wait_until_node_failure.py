import re
import sys
import time
from functools import partial, wraps
from typing import Tuple
import torch.distributed as dist
import torch.distributed.rpc as rpc
from torch.distributed.rpc import _rref_context_get_debug_info
from torch.testing._internal.common_utils import FILE_SCHEMA, TEST_WITH_TSAN
def wait_until_node_failure(rank: int, expected_error_regex: str='.*') -> str:
    """
    Loops until an RPC to the given rank fails. This is used to
    indicate that the node has failed in unit tests.
    Args:
    rank (int): Rank of the node expected to fail
    expected_error_regex (optional, str): Regex of exception message expected. Useful to ensure a specific failure
    occurs, not just any.
    """
    while True:
        try:
            rpc.rpc_sync(f'worker{rank}', noop, args=())
            time.sleep(0.1)
        except Exception as e:
            if re.search(pattern=expected_error_regex, string=str(e)):
                return str(e)