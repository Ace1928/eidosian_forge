import os
import re
import sys
from importlib import util as importlib_util
import breezy
from . import debug, errors, osutils, trace
def get_user_plugin_path():
    from breezy.bedding import config_dir
    return osutils.pathjoin(config_dir(), 'plugins')