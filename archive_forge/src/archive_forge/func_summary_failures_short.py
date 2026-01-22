import collections
import contextlib
import doctest
import functools
import importlib
import inspect
import logging
import multiprocessing
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
import unittest
from collections import defaultdict
from collections.abc import Mapping
from io import StringIO
from pathlib import Path
from typing import Callable, Dict, Iterable, Iterator, List, Optional, Union
from unittest import mock
from unittest.mock import patch
import urllib3
from transformers import logging as transformers_logging
from .integrations import (
from .integrations.deepspeed import is_deepspeed_available
from .utils import (
import asyncio  # noqa
def summary_failures_short(tr):
    reports = tr.getreports('failed')
    if not reports:
        return
    tr.write_sep('=', 'FAILURES SHORT STACK')
    for rep in reports:
        msg = tr._getfailureheadline(rep)
        tr.write_sep('_', msg, red=True, bold=True)
        longrepr = re.sub('.*_ _ _ (_ ){10,}_ _ ', '', rep.longreprtext, 0, re.M | re.S)
        tr._tw.line(longrepr)