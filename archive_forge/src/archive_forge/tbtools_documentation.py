from __future__ import annotations
import itertools
import linecache
import os
import re
import sys
import sysconfig
import traceback
import typing as t
from markupsafe import escape
from ..utils import cached_property
from .console import Console
A :class:`traceback.FrameSummary` that can evaluate code in the
    frame's namespace.
    