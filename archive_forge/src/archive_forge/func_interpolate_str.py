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
def interpolate_str(template, **data):
    s = template
    for k, v in data.items():
        key = '{%' + k + '%}'
        s = s.replace(key, v)
    return s