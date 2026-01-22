import collections
import getpass
import inspect
import os
import sys
import textwrap
import decorator
from oslo_utils import encodeutils
from oslo_utils import strutils
import prettytable
from zunclient.i18n import _
def exclusive_arg(group_name, *args, **kwargs):
    """Decorator for CLI mutually exclusive args."""

    def _decorator(func):
        required = kwargs.pop('required', None)
        add_exclusive_arg(func, group_name, required, *args, **kwargs)
        return func
    return _decorator