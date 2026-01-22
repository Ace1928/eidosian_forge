import weakref
import gc
from contextlib import contextmanager
from platform import python_implementation
@contextmanager
def gc_state(state):
    """ Context manager to set state of garbage collector to `state`

    Parameters
    ----------
    state : bool
        True for gc enabled, False for disabled

    Examples
    --------
    >>> with gc_state(False):
    ...     assert not gc.isenabled()
    >>> with gc_state(True):
    ...     assert gc.isenabled()
    """
    orig_state = gc.isenabled()
    set_gc_state(state)
    yield
    set_gc_state(orig_state)