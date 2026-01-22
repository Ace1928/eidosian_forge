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
def fp_tuple(fp):
    """
    Build a string that uniquely identifies a key
    """
    return ' '.join([fp[0], fp[1], fp[-1]])