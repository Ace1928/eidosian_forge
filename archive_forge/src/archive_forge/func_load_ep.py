import logging
import warnings
from numba.core.config import PYVERSION
def load_ep(entry_point):
    """Loads a given entry point. Warns and logs on failure.
        """
    logger.debug('Loading extension: %s', entry_point)
    try:
        func = entry_point.load()
        func()
    except Exception as e:
        msg = f"Numba extension module '{entry_point.module}' failed to load due to '{type(e).__name__}({str(e)})'."
        warnings.warn(msg, stacklevel=3)
        logger.debug('Extension loading failed for: %s', entry_point)