import importlib.util
import importlib.machinery
import os
import sys
import traceback
from gunicorn import util
from gunicorn.arbiter import Arbiter
from gunicorn.config import Config, get_default_config_file
from gunicorn import debug
def get_config_from_module_name(self, module_name):
    return vars(importlib.import_module(module_name))