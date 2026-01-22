import logging
import sys
from contextlib import contextmanager
from functools import wraps
from typing import Any, Dict, Mapping, Union
def list_keys_to_dict(key_list, callback):
    return dict.fromkeys(key_list, callback)