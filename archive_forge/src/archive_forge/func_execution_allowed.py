from contextlib import contextmanager
from jedi import debug
from jedi.inference.base_value import NO_VALUES
@contextmanager
def execution_allowed(inference_state, node):
    """
    A decorator to detect recursions in statements. In a recursion a statement
    at the same place, in the same module may not be executed two times.
    """
    pushed_nodes = inference_state.recursion_detector.pushed_nodes
    if node in pushed_nodes:
        debug.warning('catched stmt recursion: %s @%s', node, getattr(node, 'start_pos', None))
        yield False
    else:
        try:
            pushed_nodes.append(node)
            yield True
        finally:
            pushed_nodes.pop()