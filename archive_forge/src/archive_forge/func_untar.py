import importlib
import json
import time
import datetime
import os
import requests
import shutil
import hashlib
import tqdm
import math
import zipfile
import parlai.utils.logging as logging
def untar(path, fname, deleteTar=True):
    """
    Unpack the given archive file to the same directory.

    :param str path:
        The folder containing the archive. Will contain the contents.

    :param str fname:
        The filename of the archive file.

    :param bool deleteTar:
        If true, the archive will be deleted after extraction.
    """
    logging.debug(f'unpacking {fname}')
    fullpath = os.path.join(path, fname)
    shutil.unpack_archive(fullpath, path)
    if deleteTar:
        os.remove(fullpath)