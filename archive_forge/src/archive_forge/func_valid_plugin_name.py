import os
import re
import sys
from importlib import util as importlib_util
import breezy
from . import debug, errors, osutils, trace
def valid_plugin_name(name):
    return not re.search('\\.|-| ', name)