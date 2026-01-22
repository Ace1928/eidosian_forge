import logging
import os.path
def tweak_close(outer, inner):
    """Ensure that closing the `outer` stream closes the `inner` stream as well.

    Use this when your compression library's `close` method does not
    automatically close the underlying filestream.  See
    https://github.com/RaRe-Technologies/smart_open/issues/630 for an
    explanation why that is a problem for smart_open.
    """
    outer_close = outer.close

    def close_both(*args):
        nonlocal inner
        try:
            outer_close()
        finally:
            if inner:
                inner, fp = (None, inner)
                fp.close()
    outer.close = close_both