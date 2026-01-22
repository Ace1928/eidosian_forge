from __future__ import annotations
import argparse
import collections
from functools import update_wrapper
import inspect
import itertools
import operator
import os
import re
import sys
from typing import TYPE_CHECKING
import uuid
import pytest
def make_option(name, **kw):
    callback_ = kw.pop('callback', None)
    if callback_:

        class CallableAction(argparse.Action):

            def __call__(self, parser, namespace, values, option_string=None):
                callback_(option_string, values, parser)
        kw['action'] = CallableAction
    zeroarg_callback = kw.pop('zeroarg_callback', None)
    if zeroarg_callback:

        class CallableAction(argparse.Action):

            def __init__(self, option_strings, dest, default=False, required=False, help=None):
                super().__init__(option_strings=option_strings, dest=dest, nargs=0, const=True, default=default, required=required, help=help)

            def __call__(self, parser, namespace, values, option_string=None):
                zeroarg_callback(option_string, values, parser)
        kw['action'] = CallableAction
    group.addoption(name, **kw)