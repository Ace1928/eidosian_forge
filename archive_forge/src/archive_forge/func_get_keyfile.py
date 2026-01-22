import argparse
import getpass
import json
import logging
import os
import subprocess
import sys
import tempfile
import urllib.error
from urllib.parse import quote_plus
from urllib.request import Request, urlopen
import distro
from .version import VERSION
def get_keyfile(path=None):
    """Return 'path' if true, else a path to current user's authorized_keys."""
    if not path:
        if os.environ.get('HOME'):
            home = os.environ['HOME']
        else:
            home = os.path.expanduser('~' + getpass.getuser())
        path = os.path.join(home, '.ssh', 'authorized_keys')
    return path