import os
import io
import logging
import shutil
import stat
from pathlib import Path
from contextlib import contextmanager
from .. import __version__ as full_version
from ..utils import check_version, get_logger
@contextmanager
def data_over_ftp(server, fname):
    """
    Add a test data file to the test FTP server and clean it up afterwards.

    Parameters
    ----------
    server
        The ``ftpserver`` fixture provided by pytest-localftpserver.
    fname : str
        The name of a file *relative* to the test data folder of the package
        (usually just the file name, not the full path).

    Yields
    ------
    url : str
        The download URL of the data file from the test FTP server.

    """
    package_path = str(Path(__file__).parent / 'data' / fname)
    server_path = os.path.join(server.anon_root, fname)
    try:
        shutil.copyfile(package_path, server_path)
        url = f'ftp://localhost/{fname}'
        yield url
    finally:
        if os.path.exists(server_path):
            os.remove(server_path)