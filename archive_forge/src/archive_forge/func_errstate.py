import contextlib
import threading
@contextlib.contextmanager
def errstate(*, divide=None, over=None, under=None, invalid=None, linalg=None, fallback_mode=None):
    """
    TODO(hvy): Write docs.
    """
    old_state = seterr(divide=divide, over=over, under=under, invalid=invalid, linalg=linalg, fallback_mode=fallback_mode)
    try:
        yield
    finally:
        seterr(**old_state)