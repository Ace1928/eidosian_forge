import os
from breezy import trace
from breezy.rename_map import RenameMap
from breezy.tests import TestCaseWithTransport
@staticmethod
def captureNotes(cmd, *args, **kwargs):
    notes = []

    def my_note(fmt, *args):
        notes.append(fmt % args)
    old_note = trace.note
    trace.note = my_note
    try:
        result = cmd(*args, **kwargs)
    finally:
        trace.note = old_note
    return (notes, result)