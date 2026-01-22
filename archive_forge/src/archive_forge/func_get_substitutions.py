import functools
import logging
import os
import pkgutil
import re
import traceback
from oslo_utils import strutils
from zunclient import exceptions
from zunclient.i18n import _
def get_substitutions(func_name, api_version=None):
    if hasattr(func_name, '__id__'):
        func_name = func_name.__id__
    substitutions = _SUBSTITUTIONS.get(func_name, [])
    if api_version and (not api_version.is_null()):
        return [m for m in substitutions if api_version.matches(m.start_version, m.end_version)]
    return sorted(substitutions, key=lambda m: m.start_version)