import codecs
import functools
import os
import pickle
import re
import sys
import textwrap
import zipfile
from abc import ABCMeta, abstractmethod
from gzip import WRITE as GZ_WRITE
from gzip import GzipFile
from io import BytesIO, TextIOWrapper
from urllib.request import url2pathname, urlopen
from nltk import grammar, sem
from nltk.compat import add_py3_data, py3_data
from nltk.internals import deprecated
def split_resource_url(resource_url):
    """
    Splits a resource url into "<protocol>:<path>".

    >>> windows = sys.platform.startswith('win')
    >>> split_resource_url('nltk:home/nltk')
    ('nltk', 'home/nltk')
    >>> split_resource_url('nltk:/home/nltk')
    ('nltk', '/home/nltk')
    >>> split_resource_url('file:/home/nltk')
    ('file', '/home/nltk')
    >>> split_resource_url('file:///home/nltk')
    ('file', '/home/nltk')
    >>> split_resource_url('file:///C:/home/nltk')
    ('file', '/C:/home/nltk')
    """
    protocol, path_ = resource_url.split(':', 1)
    if protocol == 'nltk':
        pass
    elif protocol == 'file':
        if path_.startswith('/'):
            path_ = '/' + path_.lstrip('/')
    else:
        path_ = re.sub('^/{0,2}', '', path_)
    return (protocol, path_)