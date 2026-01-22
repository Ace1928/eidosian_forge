from __future__ import absolute_import, division, print_function
import sys
import logging
import contextlib
import copy
import os
from future.utils import PY2, PY3
def cache_py2_modules():
    """
    Currently this function is unneeded, as we are not attempting to provide import hooks
    for modules with ambiguous names: email, urllib, pickle.
    """
    if len(sys.py2_modules) != 0:
        return
    assert not detect_hooks()
    import urllib
    sys.py2_modules['urllib'] = urllib
    import email
    sys.py2_modules['email'] = email
    import pickle
    sys.py2_modules['pickle'] = pickle