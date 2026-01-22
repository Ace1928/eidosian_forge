import os
import subprocess
import contextlib
import functools
import tempfile
import shutil
import operator
import warnings
def infer_compression(url):
    """
    Given a URL or filename, infer the compression code for tar.

    >>> infer_compression('http://foo/bar.tar.gz')
    'z'
    >>> infer_compression('http://foo/bar.tgz')
    'z'
    >>> infer_compression('file.bz')
    'j'
    >>> infer_compression('file.xz')
    'J'
    """
    compression_indicator = url[-2:]
    mapping = dict(gz='z', bz='j', xz='J')
    return mapping.get(compression_indicator, 'z')