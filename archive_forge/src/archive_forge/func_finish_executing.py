from __future__ import print_function
import gdb
import os
import locale
import sys
import sys
import libpython
import re
import warnings
import tempfile
import functools
import textwrap
import itertools
import traceback
def finish_executing(self, result):
    """
        After doing some kind of code running in the inferior, print the line
        of source code or the result of the last executed gdb command (passed
        in as the `result` argument).
        """
    output_on_halt, output_always = self.filter_output(result)
    if self.stopped():
        print(output_always)
        print(output_on_halt)
    else:
        frame = gdb.selected_frame()
        source_line = self.lang_info.get_source_line(frame)
        if self.lang_info.is_relevant_function(frame):
            raised_exception = self.lang_info.exc_info(frame)
            if raised_exception:
                print(raised_exception)
        if source_line:
            if output_always.rstrip():
                print(output_always.rstrip())
            print(source_line)
        else:
            print(result)