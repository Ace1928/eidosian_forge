from collections import namedtuple
import dis
from functools import partial
import itertools
import os.path
import sys
from _pydevd_frame_eval.vendored import bytecode
from _pydevd_frame_eval.vendored.bytecode.instr import Instr, Label
from _pydev_bundle import pydev_log
from _pydevd_frame_eval.pydevd_frame_tracing import _pydev_stop_at_break, _pydev_needs_stop_at_break

    Inserts pydevd programmatic breaks into the code (at the given lines).

    :param breakpoint_lines: set with the lines where we should add breakpoints.
    :return: tuple(boolean flag whether insertion was successful, modified code).
    