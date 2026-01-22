import functools
import itertools
import os
import shutil
import subprocess
import sys
import textwrap
import threading
import time
import warnings
import zipfile
from hashlib import md5
from xml.etree import ElementTree
from urllib.error import HTTPError, URLError
from urllib.request import urlopen
import nltk
def incr_download(self, info_or_id, download_dir=None, force=False):
    if download_dir is None:
        download_dir = self._download_dir
        yield SelectDownloadDirMessage(download_dir)
    if isinstance(info_or_id, (list, tuple)):
        yield from self._download_list(info_or_id, download_dir, force)
        return
    try:
        info = self._info_or_id(info_or_id)
    except (OSError, ValueError) as e:
        yield ErrorMessage(None, f'Error loading {info_or_id}: {e}')
        return
    if isinstance(info, Collection):
        yield StartCollectionMessage(info)
        yield from self.incr_download(info.children, download_dir, force)
        yield FinishCollectionMessage(info)
    else:
        yield from self._download_package(info, download_dir, force)