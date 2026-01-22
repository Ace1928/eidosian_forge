from __future__ import print_function
import os
import sys
import argparse
import json
import atexit
from ansible.module_utils.six.moves import configparser
from ansible.module_utils.compat.version import LooseVersion
from ansible.module_utils.urls import Request
from ansible.module_utils.six.moves.urllib.error import URLError, HTTPError
def hoststub(self):
    return {'hosts': []}