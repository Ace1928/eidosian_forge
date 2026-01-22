import argparse
import collections
import functools
import glob
import inspect
import itertools
import os
import re
import subprocess
import sys
import threading
import unicodedata
from enum import (
from typing import (
from . import (
from .argparse_custom import (
def quote_string_if_needed(arg: str) -> str:
    """Quote a string if it contains spaces and isn't already quoted"""
    if is_quoted(arg) or ' ' not in arg:
        return arg
    return quote_string(arg)