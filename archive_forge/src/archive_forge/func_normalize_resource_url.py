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
def normalize_resource_url(resource_url):
    """
    Normalizes a resource url

    >>> windows = sys.platform.startswith('win')
    >>> os.path.normpath(split_resource_url(normalize_resource_url('file:grammar.fcfg'))[1]) == \\
    ... ('\\\\' if windows else '') + os.path.abspath(os.path.join(os.curdir, 'grammar.fcfg'))
    True
    >>> not windows or normalize_resource_url('file:C:/dir/file') == 'file:///C:/dir/file'
    True
    >>> not windows or normalize_resource_url('file:C:\\\\dir\\\\file') == 'file:///C:/dir/file'
    True
    >>> not windows or normalize_resource_url('file:C:\\\\dir/file') == 'file:///C:/dir/file'
    True
    >>> not windows or normalize_resource_url('file://C:/dir/file') == 'file:///C:/dir/file'
    True
    >>> not windows or normalize_resource_url('file:////C:/dir/file') == 'file:///C:/dir/file'
    True
    >>> not windows or normalize_resource_url('nltk:C:/dir/file') == 'file:///C:/dir/file'
    True
    >>> not windows or normalize_resource_url('nltk:C:\\\\dir\\\\file') == 'file:///C:/dir/file'
    True
    >>> windows or normalize_resource_url('file:/dir/file/toy.cfg') == 'file:///dir/file/toy.cfg'
    True
    >>> normalize_resource_url('nltk:home/nltk')
    'nltk:home/nltk'
    >>> windows or normalize_resource_url('nltk:/home/nltk') == 'file:///home/nltk'
    True
    >>> normalize_resource_url('https://example.com/dir/file')
    'https://example.com/dir/file'
    >>> normalize_resource_url('dir/file')
    'nltk:dir/file'
    """
    try:
        protocol, name = split_resource_url(resource_url)
    except ValueError:
        protocol = 'nltk'
        name = resource_url
    if protocol == 'nltk' and os.path.isabs(name):
        protocol = 'file://'
        name = normalize_resource_name(name, False, None)
    elif protocol == 'file':
        protocol = 'file://'
        name = normalize_resource_name(name, False, None)
    elif protocol == 'nltk':
        protocol = 'nltk:'
        name = normalize_resource_name(name, True)
    else:
        protocol += '://'
    return ''.join([protocol, name])