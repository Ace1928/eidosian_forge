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
def read_keyfile():
    """
    Locate key file, read the current state, return lines in a list
    """
    keyfile = get_keyfile(parser.options.output)
    if keyfile == '-' or not os.path.exists(keyfile):
        lines = []
    else:
        try:
            with open(keyfile, 'r') as fp:
                lines = fp.readlines()
        except OSError:
            die('Could not read authorized key file [%s]' % keyfile)
    return lines