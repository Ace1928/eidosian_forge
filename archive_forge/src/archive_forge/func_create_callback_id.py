import shlex
import sys
import uuid
import hashlib
import collections
import subprocess
import logging
import io
import json
import secrets
import string
import inspect
from html import escape
from functools import wraps
from typing import Union
from dash.types import RendererHooks
def create_callback_id(output, inputs):
    hashed_inputs = None

    def _concat(x):
        nonlocal hashed_inputs
        _id = x.component_id_str().replace('.', '\\.') + '.' + x.component_property
        if x.allow_duplicate:
            if not hashed_inputs:
                hashed_inputs = hashlib.md5('.'.join((str(x) for x in inputs)).encode('utf-8')).hexdigest()
            _id += f'@{hashed_inputs}'
        return _id
    if isinstance(output, (list, tuple)):
        return '..' + '...'.join((_concat(x) for x in output)) + '..'
    return _concat(output)