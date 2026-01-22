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
def clear_status_cache(self, id=None):
    if id is None:
        self._status_cache.clear()
    else:
        self._status_cache.pop(id, None)