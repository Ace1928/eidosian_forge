from __future__ import annotations
import atexit
import dis
import itertools
import sys
import threading
from types import FrameType, ModuleType
from typing import Any, Callable, Set, cast
from coverage import env
from coverage.types import (
Return a dictionary of statistics, or None.