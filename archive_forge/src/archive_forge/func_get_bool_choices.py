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
def get_bool_choices(_) -> List[str]:
    """Used to tab complete lowercase boolean values"""
    return ['true', 'false']