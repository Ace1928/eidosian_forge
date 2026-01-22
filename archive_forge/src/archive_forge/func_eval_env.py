import importlib.util
import os
import warnings
from functools import wraps
from typing import Optional
def eval_env(var, default):
    """Check if environment varable has True-y value"""
    if var not in os.environ:
        return default
    val = os.environ.get(var, '0')
    trues = ['1', 'true', 'TRUE', 'on', 'ON', 'yes', 'YES']
    falses = ['0', 'false', 'FALSE', 'off', 'OFF', 'no', 'NO']
    if val in trues:
        return True
    if val not in falses:
        raise RuntimeError(f'Unexpected environment variable value `{var}={val}`. Expected one of {trues + falses}')
    return False