import os
import io
import logging
import shutil
import stat
from pathlib import Path
from contextlib import contextmanager
from .. import __version__ as full_version
from ..utils import check_version, get_logger
def pooch_test_dataverse_url():
    """
    Get the base URL for the test data stored on a DataVerse instance.

    Returns
    -------
    url
        The URL for pooch's test data.
    """
    url = 'doi:10.11588/data/TKCFEF/'
    return url