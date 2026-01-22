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
def unique_file_name(url):
    """
    Create a unique file name based on the given URL.

    The file name will be unique to the URL by prepending the name with the MD5
    hash (hex digest) of the URL. The name will also include the last portion
    of the URL.

    The format will be: ``{md5}-{filename}.{ext}``

    The file name will be cropped so that the entire name (including the hash)
    is less than 255 characters long (the limit on most file systems).

    Parameters
    ----------
    url : str
        The URL with a file name at the end.

    Returns
    -------
    fname : str
        The file name, unique to this URL.

    Examples
    --------

    >>> print(unique_file_name("https://www.some-server.org/2020/data.txt"))
    02ddee027ce5ebb3d7059fb23d210604-data.txt
    >>> print(unique_file_name("https://www.some-server.org/2019/data.txt"))
    9780092867b497fca6fc87d8308f1025-data.txt
    >>> print(unique_file_name("https://www.some-server.org/2020/data.txt.gz"))
    181a9d52e908219c2076f55145d6a344-data.txt.gz

    """
    md5 = hashlib.md5(url.encode()).hexdigest()
    fname = parse_url(url)['path'].split('/')[-1]
    fname = fname[-(255 - len(md5) - 1):]
    unique_name = f'{md5}-{fname}'
    return unique_name