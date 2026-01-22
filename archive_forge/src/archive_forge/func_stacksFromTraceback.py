import sys, traceback
from ..Qt import QtWidgets, QtGui
def stacksFromTraceback(tb, lastFrame=None):
    """Return (text, stack_frame) for a traceback and the stack preceding it

    If *lastFrame* is given and present in the stack, then the stack is truncated 
    at that frame.
    """
    stack = stackFromFrame(tb.tb_frame.f_back if tb is not None else lastFrame)
    if tb is None:
        return (stack, [])
    lines = traceback.format_tb(tb)
    frames = []
    while True:
        frames.append(tb.tb_frame)
        if tb.tb_next is None or tb.tb_frame is lastFrame:
            break
        tb = tb.tb_next
    return (stack, list(zip(lines[:len(frames)], frames)))