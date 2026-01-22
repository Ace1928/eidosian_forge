import os
import re
import sys
from importlib import util as importlib_util
import breezy
from . import debug, errors, osutils, trace
def sanitise_plugin_name(name):
    sanitised_name = re.sub('[-. ]', '_', name)
    if sanitised_name.startswith('brz_'):
        sanitised_name = sanitised_name[len('brz_'):]
    return sanitised_name