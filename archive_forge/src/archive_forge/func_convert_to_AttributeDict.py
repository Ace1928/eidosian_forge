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
def convert_to_AttributeDict(nested_list):
    new_dict = []
    for i in nested_list:
        if isinstance(i, dict):
            new_dict.append(AttributeDict(i))
        else:
            new_dict.append([AttributeDict(ii) for ii in i])
    return new_dict