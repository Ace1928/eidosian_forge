from __future__ import absolute_import, division, print_function
import json
import os
import sys
import uuid
import random
import re
import socket
from datetime import datetime
from ssl import SSLError
from http.client import RemoteDisconnected
from time import time
from ansible.module_utils.urls import open_url
from ansible.module_utils.six.moves.urllib.error import (
from .constants import (
from .version import CURRENT_COLL_VERSION
def determine_environment():
    for key in CICD_ENV:
        env = os.getenv(key)
        if env:
            if key == 'CI_NAME' and env == 'codeship':
                return CICD_ENV[key]
            if key == 'CI_NAME' and env != 'codeship':
                return None
            return CICD_ENV[key]