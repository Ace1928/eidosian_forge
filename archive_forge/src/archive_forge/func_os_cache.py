import logging
import os
import tempfile
import hashlib
from pathlib import Path
from urllib.parse import urlsplit
from contextlib import contextmanager
import warnings
import platformdirs
from packaging.version import Version
def os_cache(project):
    """
    Default cache location based on the operating system.

    The folder locations are defined by the ``platformdirs``  package
    using the ``user_cache_dir`` function.
    Usually, the locations will be following (see the
    `platformdirs documentation <https://platformdirs.readthedocs.io>`__):

    * Mac: ``~/Library/Caches/<AppName>``
    * Unix: ``~/.cache/<AppName>`` or the value of the ``XDG_CACHE_HOME``
      environment variable, if defined.
    * Windows: ``C:\\Users\\<user>\\AppData\\Local\\<AppAuthor>\\<AppName>\\Cache``

    Parameters
    ----------
    project : str
        The project name.

    Returns
    -------
    cache_path : :class:`pathlib.Path`
        The default location for the data cache. User directories (``'~'``) are
        not expanded.

    """
    return Path(platformdirs.user_cache_dir(project))