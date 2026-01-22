from contextlib import contextmanager
import os
import shutil
import sys
import time
import logging
import inspect
import pprint
import subprocess
import textwrap
@contextmanager
def limit_logging(max_lvl=logging.CRITICAL):
    """Contextmanager for silencing logging messages.

    Examples
    --------
    >>> with limit_logging():
    ...     logger.info("you won't see this...")  # doctest: +SKIP

    """
    _ori = logging.root.manager.disable
    logging.disable(max_lvl)
    try:
        yield
    finally:
        logging.disable(_ori)