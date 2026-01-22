import traceback
from contextlib import contextmanager
from typing import List, Any, Dict
from ._compatibility import compatibility
@compatibility(is_backward_compatible=False)
def reset_grad_fn_seq_nr():
    global current_meta
    if should_preserve_node_meta:
        if current_meta['prev_grad_fn_seq_nr'] is None:
            assert current_meta['prev_in_grad_fn'] is None
            del current_meta['grad_fn_seq_nr']
            del current_meta['in_grad_fn']
        current_meta['grad_fn_seq_nr'] = current_meta['prev_grad_fn_seq_nr']
        current_meta['in_grad_fn'] = current_meta['prev_in_grad_fn']