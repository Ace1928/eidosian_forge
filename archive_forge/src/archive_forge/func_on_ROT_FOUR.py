from _pydev_bundle import pydev_log
from types import CodeType
from _pydevd_frame_eval.vendored.bytecode.instr import _Variable
from _pydevd_frame_eval.vendored import bytecode
from _pydevd_frame_eval.vendored.bytecode import cfg as bytecode_cfg
import dis
import opcode as _opcode
from _pydevd_bundle.pydevd_constants import KeyifyList, DebugInfoHolder, IS_PY311_OR_GREATER
from bisect import bisect
from collections import deque
def on_ROT_FOUR(self, instr):
    try:
        p0 = self._stack.pop()
    except IndexError:
        return
    try:
        p1 = self._stack.pop()
    except:
        self._stack.append(p0)
        return
    try:
        p2 = self._stack.pop()
    except:
        self._stack.append(p0)
        self._stack.append(p1)
        return
    try:
        p3 = self._stack.pop()
    except:
        self._stack.append(p0)
        self._stack.append(p1)
        self._stack.append(p2)
        return
    self._stack.append(p0)
    self._stack.append(p1)
    self._stack.append(p2)
    self._stack.append(p3)