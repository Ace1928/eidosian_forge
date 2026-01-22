import os
import io
import re
import sys
import cmd
import bdb
import dis
import code
import glob
import pprint
import signal
import inspect
import tokenize
import functools
import traceback
import linecache
from typing import Union
def bp_commands(self, frame):
    """Call every command that was set for the current active breakpoint
        (if there is one).

        Returns True if the normal interaction function must be called,
        False otherwise."""
    if getattr(self, 'currentbp', False) and self.currentbp in self.commands:
        currentbp = self.currentbp
        self.currentbp = 0
        lastcmd_back = self.lastcmd
        self.setup(frame, None)
        for line in self.commands[currentbp]:
            self.onecmd(line)
        self.lastcmd = lastcmd_back
        if not self.commands_silent[currentbp]:
            self.print_stack_entry(self.stack[self.curindex])
        if self.commands_doprompt[currentbp]:
            self._cmdloop()
        self.forget()
        return
    return 1