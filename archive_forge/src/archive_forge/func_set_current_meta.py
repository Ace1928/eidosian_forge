import traceback
from contextlib import contextmanager
from typing import List, Any, Dict
from ._compatibility import compatibility
@compatibility(is_backward_compatible=False)
@contextmanager
def set_current_meta(node):
    global current_meta
    if should_preserve_node_meta and node.meta:
        saved_meta = current_meta
        try:
            current_meta = node.meta.copy()
            if 'from_node' not in current_meta:
                current_meta['from_node'] = [(node.name, node.target)]
            elif current_meta['from_node'][-1][0] != node.name:
                current_meta['from_node'].append((node.name, node.target))
            yield
        finally:
            current_meta = saved_meta
    else:
        yield