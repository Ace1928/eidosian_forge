import traceback
from contextlib import contextmanager
from typing import List, Any, Dict
from ._compatibility import compatibility
@compatibility(is_backward_compatible=False)
def set_stack_trace(stack: List[str]):
    global current_meta
    if should_preserve_node_meta and stack:
        current_meta['stack_trace'] = ''.join(stack)